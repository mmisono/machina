"""
Sampler using ray
"""

import copy
import time

import ray
import gym
import numpy as np
import torch
import torch.multiprocessing as mp

from machina.utils import cpu_mode


LARGE_NUMBER = 100000000

"""
- Overview

EpiSampler <----- Worker  <----- Env
              |              |-- Env
              |              |-- Env
              |
              |-- Worker <------ Env
                             |-- Env
                             |-- Env

- Each env return one episode at a time
- Each worker return num_epis episodes at a time
"""


class Env:
    def __init__(self, pol, env, seed, prepro):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        self.prepro = prepro
        self.pol = pol
        self.env = env

    @classmethod
    def as_remote(cls):
        return ray.remote(cls)

    def set_pol(self, pol):
        self.pol = pol
        self.pol.eval()

    def set_pol_state(self, state_dict):
        self.pol.load_state_dict(state_dict)
        self.pol.eval()

    def one_epi(self, deterministic=False):
        with cpu_mode():
            obs = []
            acs = []
            rews = []
            dones = []
            a_is = []
            e_is = []
            o = self.env.reset()
            self.pol.reset()
            done = False
            epi_length = 0
            while not done:
                o = self.prepro(o)
                if not deterministic:
                    ac_real, ac, a_i = self.pol(
                        torch.tensor(o, dtype=torch.float))
                else:
                    ac_real, ac, a_i = self.pol.deterministic_ac_real(
                        torch.tensor(o, dtype=torch.float))
                ac_real = ac_real.reshape(self.pol.action_space.shape)
                next_o, r, done, e_i = self.env.step(np.array(ac_real))
                obs.append(o)
                rews.append(r)
                dones.append(done)
                acs.append(ac.squeeze().detach().cpu(
                ).numpy().reshape(self.pol.action_space.shape))
                _a_i = dict()
                for key in a_i.keys():
                    if a_i[key] is None:
                        continue
                    if isinstance(a_i[key], tuple):
                        _a_i[key] = tuple([h.squeeze().detach().cpu().numpy()
                                           for h in a_i[key]])
                    else:
                        _a_i[key] = a_i[key].squeeze().detach(
                        ).cpu().numpy().reshape(self.pol.a_i_shape)
                a_i = _a_i
                a_is.append(a_i)
                e_is.append(e_i)
                epi_length += 1
                if done:
                    break
                o = next_o

            return epi_length, dict(
                obs=np.array(obs, dtype='float32'),
                acs=np.array(acs, dtype='float32'),
                rews=np.array(rews, dtype='float32'),
                dones=np.array(dones, dtype='float32'),
                a_is=dict([(key, np.array([a_i[key] for a_i in a_is], dtype='float32'))
                           for key in a_is[0].keys()]),
                e_is=dict([(key, np.array([e_i[key] for e_i in e_is], dtype='float32'))
                           for key in e_is[0].keys()])
            )


class Worker:
    def __init__(self, pol, env, seed, num_envs, prepro=None):
        if prepro is None:
            def prepro(x): return x

        self.envs = [Env.as_remote().remote(pol, env, seed + i, prepro)
                     for i in range(num_envs)]
        self.set_pol(pol)

    @classmethod
    def as_remote(cls):
        return ray.remote(cls)

    def set_pol(self, pol):
        for env in self.envs:
            env.set_pol.remote(pol)

    def set_pol_state(self, state_dict):
        for env in self.envs:
            env.set_pol_state.remote(state_dict)

    def sample(self, num_epis, deterministic):
        """Collect num_epis of episodes
        """
        pending = {env.one_epi.remote(deterministic): env for env in self.envs}

        epis = []
        n_epis = 0
        while pending:
            ready, _ = ray.wait(list(pending))
            for obj_id in ready:
                env = pending.pop(obj_id)
                epis.append(obj_id)
                n_epis += 1
                if (n_epis + len(pending)) < num_epis:
                    pending[env.one_epi.remote(deterministic)] = env
        return epis


class EpiSampler(object):
    """
    A sampler which sample episodes.

    Parameters
    ----------
    env : gym.Env
    pol : Pol
    num_parallel : int
        Number of processes
    prepro : Prepro
    seed : int
    """

    def __init__(self, pol, env, num_workers=8, num_envs=1, num_batch_epi=1, prepro=None, seed=256):
        pol = copy.deepcopy(pol)
        pol.to('cpu')

        pol = ray.put(pol)
        env = ray.put(env)

        self.workers = [Worker.as_remote().remote(pol, env, seed + i * num_envs, num_envs, prepro)
                        for i in range(num_workers)]
        self.num_batch_epi = num_batch_epi

    def set_pol(self, pol):
        if not isinstance(pol, ray.ObjectID):
            pol = ray.put(pol)
        for w in self.workers:
            w.set_pol.remote(pol)

    def set_pol_state(self, state_dict):
        if not isinstance(state_dict, ray.ObjectID):
            state_dict = ray.put(state_dict)
        for w in self.workers:
            w.set_pol_state.remote(state_dict)

    def sample(self, max_epis=None, max_steps=None, deterministic=False):
        """
        Switch on sampling processes.

        Parameters
        ----------
        pol : Pol
        max_epis : int or None
            maximum episodes of episodes.
            If None, this value is ignored.
        max_steps : int or None
            maximum steps of episodes
            If None, this value is ignored.
        deterministic : bool

        Returns
        -------
        epis : list of dict
            Sampled epis.

        Raises
        ------
        ValueError
            If max_steps and max_epis are botch None.
        """

        if max_epis is None and max_steps is None:
            raise ValueError(
                'Either max_epis or max_steps needs not to be None')
        max_epis = max_epis if max_epis is not None else LARGE_NUMBER
        max_steps = max_steps if max_steps is not None else LARGE_NUMBER

        epis = []
        n_steps = 0
        n_epis = 0

        pending = {w.sample.remote(
            self.num_batch_epi, deterministic): w for w in self.workers}

        while pending:
            ready, _ = ray.wait(list(pending))
            for obj_id in ready:
                worker = pending.pop(obj_id)
                results = ray.get(obj_id)
                for (l, epi) in ray.get(results):
                    epis.append(epi)
                    n_steps += l
                    n_epis += 1
                if n_steps < max_steps and (n_epis + len(pending)) < max_epis:
                    pending[worker.sample.remote(
                        self.num_batch_epi, deterministic)] = worker

        return epis
