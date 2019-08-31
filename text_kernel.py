import node_role_types as nrt
import scipy.sparse as ss
import multiprocessing as mp
import parallel_computing as pc
import constants_absolute_path as cap
import numpy as np
import time
import pickle
import math
import numpy.linalg as npl
import save_sparse_scipy_matrices
import scipy.sparse.linalg as ssl
import scipy.special
import numpy.random as npr
from numpy.random import RandomState
import random as rnd
import json
import datetime
import scipy.io
import config_kernel
from config import *
import copy
import scipy.linalg as spl
import nltk
import nltk.stem as nltk_stem
from nltk.corpus import wordnet
import constant_dialogue_response_prefixes as cdrp
import nonascii_text_processing as ntp
import twitter_dataset_preprocessed_fr_dialog_modeling as tdpdm
import pyximport; pyximport.install()
import subsequence_kernel_cython as skc
import cython_dialogue_modeling as cdm
import cython_wordvec_kernel_dialogue_modeling as cwkdm
import parallel_computing_wrapper as pcw
import string as st


class TextKernel:

    def __init__(self,
                 num_cores,
                 is_edge_label_sim,
                 lamb, cs, p,
                 is_normalize_kernel_fr_hashcodes,
                 is_wordvec_large=False,
                 p_weights=None,
    ):

        self.__is_debug__ = False
        self.__is_coarse_debug__ = False

        self.num_cores = num_cores

        # kernel computation related settings
        self.__wordvec_buffer__ = {}
        self.__kernel_buffer_node_edge_tuple__ = {}
        self.__kernel_sigma__ = None

        self.__global_sigma_f__ = 1

        self.__matern_l__ = 1
        # keep mu corresponds to differentiability of the covariance function.
        self.__matern_nu__ = 10

        self.__is_sparsity_proximal_operator__ = False
        self.__sparsity_proximal_operator_node_label_kernel__ = 0
        self.__sparsity_proximal_operator_edge_label_kernel__ = 0
        self.__sparsity_proximal_operator_wordvec_kernel__ = 0

        # for parallel computing with a high number of cores, buffer slows down kernel matrix computations so. so one can switch use of different buffer used.
        # make sure to use this flag whenever implementing any buffer or a uncommenting an old one.
        self.__is_buffer__ = False
        self.__is_wordvec_kernel_false__ = True

        self.__num_data_size_scales__ = 1
        self.__scale_mul__ = 10

        self.__word_vectors_map__ = None

        self.__is_node_label_matrix__ = False
        self.__is_edge_label_matrix__ = False

        self.set_parameters(is_edge_label_sim=is_edge_label_sim,
                            lamb=lamb, cs=cs, p=p,
                            is_normalize_kernel_fr_hashcodes=is_normalize_kernel_fr_hashcodes,
                            is_wordvec_large=is_wordvec_large,
                            p_weights=p_weights)
        self.__reset_buffers__()

    def set_parameters(self,
                       is_edge_label_sim,
                       lamb, cs, p,
                       is_normalize_kernel_fr_hashcodes,
                       is_wordvec_large, p_weights):

        self.__is_coarse_debug__ = False

        self.is_error_analysis = False

        self.__is_buffer__ = True
        self.__is_wordvec_kernel_false__ = True

        if not self.__is_wordvec_kernel_false__:
            self.glove_wordvec_map = self.load_glove_model()

        self.__is_edge_label_sigma_f__ = False
        self.__is_node_label_sigma_f__ = False

        # self.gloveFile = './GoogleNews-vectors-negative300.txt'
        # self.gloveFile = './glove.twitter.27B.200d.txt'

        if not is_wordvec_large:
            self.gloveFile = './glove.6B.100d.txt'
            self.wordvec_length = 100
        else:
            self.gloveFile = './glove.840B.300d.txt'
            self.wordvec_length = 300

        self.__kernel_sigma__ = np.ones(self.wordvec_length)
        print 'self.__kernel_sigma__', self.__kernel_sigma__.tolist()
        self.__is_sparse_kernel__ = True
        print 'self.__is_sparse_kernel__', self.__is_sparse_kernel__
        self.__sparse_kernel_v__ = 1
        print 'self.__sparse_kernel_v__', self.__sparse_kernel_v__

        self.__reset_buffers__(is_wordvec_buffer_reset=True)

        self.load_glove_model()

        self.is_edge_label_sim = False
        self.is_wordnet_sim = False
        self.__is_wordvec_kernel_false__ = False

        # if self.is_edge_label_sim:
        #     self.__lamb__ = 0.8
        #     self.__sparse_kernel_threshold__ = 0.0
        #     # self.__sparse_kernel_threshold__ = 0.4
        #     self.__p__ = 16
        # else:
        #     # self.__lamb__ = 0.8
        #     # self.__lamb__ = 0.95
        #     self.__lamb__ = 0.99
        #     # self.__sparse_kernel_threshold__ = 0.75
        #     # self.__p__ = 16
        #     # self.__sparse_kernel_threshold__ = 0.0
        #     # self.__sparse_kernel_threshold__ = 0.5
        #     self.__sparse_kernel_threshold__ = 0.65
        #     self.__p__ = 16
        #     # self.__p__ = 8

        self.is_edge_label_sim = is_edge_label_sim
        self.__lamb__ = lamb
        self.__sparse_kernel_threshold__ = cs
        self.__p__ = p

        if p_weights is not None:
            assert len(p_weights) == (self.__p__+1)
            self.__p_weights__ = np.array(p_weights)
            self.__p_weights__ /= (self.__p_weights__.sum() + 1e-100)
        else:
            self.__compute_uniform_weight_subsequence_length__()
            # self.__compute_weight_subsequences_length__(weight_power_factor=1)
            # self.__compute_weight_subsequences_length__(weight_power_factor=2)

        self.is_normalize_kernel_fr_hashcodes = is_normalize_kernel_fr_hashcodes

        print 'self.__lamb__', self.__lamb__
        print 'self.__global_sigma_f__', self.__global_sigma_f__
        print 'self.__sparsity_proximal_operator_wordvec_kernel__', self.__sparsity_proximal_operator_wordvec_kernel__
        print 'self.__p_weights__', self.__p_weights__

    def __reset_buffers__(self, is_wordvec_buffer_reset=False):

        self.__kernel_buffer_node_edge_tuple__ = {}
        # self.__wordvec_transformed_buffer__ = {}
        self.__kernel_buffer_edge_tuple__ = {}
        self.__kernel_buffer_node_tuple__ = {}
        self.__randomvec_buffer__ = {}

        if is_wordvec_buffer_reset:
            self.__wordvec_buffer__ = {}

    def __get_random_vector__(self, label):

        assert label is not None

        if (not hasattr(self, '__randomvec_buffer__')) or (self.__randomvec_buffer__ is None):
            print 'initializing random vectors buffer'
            self.__randomvec_buffer__ = {}

        if label not in self.__randomvec_buffer__:
            print 'warning: random vector'
            random_i_vector = npr.random(self.random_word_vec_len)
            self.__randomvec_buffer__[label] = random_i_vector
        else:
            random_i_vector = self.__randomvec_buffer__[label]

        return random_i_vector

    def __compute_weight_subsequences_length__(self, weight_power_factor=1):
        # Even if there is no gap, long sub-sequence matches get less weight due to lambda decay.
        # That is not appropriate. So, computing weights as per value of p.
        # self.__p_weights__ = np.array([self.__lamb__**x for x in xrange(self.__p__+1)])
        # even more stronger weights on
        self.__p_weights__ = np.array([(self.__lamb__**(x*weight_power_factor)) for x in xrange(self.__p__+1)])
        self.__p_weights__ = 1/self.__p_weights__
        self.__p_weights__ /= self.__p_weights__.sum()

    def __compute_uniform_weight_subsequence_length__(self,):
        self.__p_weights__ = np.ones(self.__p__ + 1)
        self.__p_weights__ /= (self.__p_weights__.sum() + 1e-100)

    def __sparse_func__(self, k, threshold):
        k_sparse = 1-((1-k)/(1-threshold))
        # sparse component
        if k_sparse < 0:
            k_sparse = 0
        k_sparse = k_sparse**self.__sparse_kernel_v__
        return k_sparse

    def __compute_kernel_frm_cosine_similarity__(self, normalized_cs_ij):
        # kernel computation
        kij = math.exp(normalized_cs_ij-1)
        kij_sparse = self.__sparse_func__(normalized_cs_ij, self.__sparse_kernel_threshold__)
        kij *= kij_sparse
        return kij

    def __matern_kernel__(self, r):
        pow2_term_nu = 2**(1-self.__matern_nu__)
        gamma_term_nu = scipy.special.gamma(self.__matern_nu__)
        root_term_nu = math.sqrt(2*self.__matern_nu__)
        raise NotImplementedError

    def __compute_kernel_for_cs__(self, Q_ij):
        # kernel computation
        kij = math.exp(Q_ij-1)
        # print 'kij', kij

        if self.__is_sparse_kernel__:
            # print 'kij', kij
            # print 'warning: no sparsity in the kernel implemented yet.'
            kij_sparse = self.__sparse_func__(Q_ij, self.__sparse_kernel_threshold__)

            # if kij_sparse > 0:
            #     print 'kij_sparse', kij_sparse

            kij *= kij_sparse
            kij_sparse = None

        return kij

    def __compute_kernel_word_vec_pairwise__(self, word_i_vector, word_j_vector):
        if (word_i_vector is None) or (word_j_vector is None):
            return 0

        Q_ij = word_i_vector.dot(word_j_vector.T)
        # print 'Q_ij', Q_ij

        return self.__compute_kernel_for_cs__(Q_ij=Q_ij)

    def load_glove_model(self):

        gloveFile = self.gloveFile

        start_time = time.time()
        print "Loading Glove Model ..."
        f = open(gloveFile, 'r')
        model = {}

        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            # print 'word', word
            embedding = [float(val) for val in splitLine[1:]]
            embedding = np.array(embedding)
            # print 'embedding', embedding
            model[word] = embedding
        print "Done.", len(model), " words loaded!"
        print 'Time to load was ', time.time() - start_time
        start_time = None

        self.glove_wordvec_map = model

    def preprocess_word_fr_cs(self, curr_word):
        curr_word = curr_word.lower()
        curr_word = curr_word.strip('.')
        curr_word = curr_word.strip()
        curr_word = curr_word.strip('\'')
        curr_word = ntp.remove_non_ascii(curr_word)
        return curr_word

    def __search_object_for_word_from_map__(self,
                                            word_str,
                                            objects_map,
                                            is_simple_search=False):

        if word_str in objects_map:
            curr_object = objects_map[word_str]
            return curr_object

        if not is_simple_search:
            word_str_lower = word_str.lower()
            if word_str_lower in objects_map:
                curr_object = objects_map[word_str_lower]
                return curr_object
            word_str_lower = None

            word_str_upper = word_str.upper()
            if word_str_upper in objects_map:
                curr_object = objects_map[word_str_upper]
                return curr_object
            word_str_upper = None

            word_str_formatted = word_str.strip('.')
            word_str_formatted = word_str_formatted.strip()
            word_str_formatted = word_str_formatted.strip('\'')
            word_str_formatted = ntp.remove_non_ascii(word_str_formatted)
            word_str_formatted = word_str_formatted
            if word_str_formatted in objects_map:
                curr_object = objects_map[word_str_formatted]
                return curr_object
            word_str_formatted = None

            # # this operation is very expensive, must avoid it, so commented  (can take fraction of second for one single call)
            # word_str_spelled = autocorrect.spell(word_str)
            # if word_str_spelled in objects_map:
            #     curr_object = objects_map[word_str_spelled]
            #     return curr_object
            # word_str_spelled = None

        return None

    def get_wordvector_wrapper(self, word_str):
        assert self.glove_wordvec_map is not None
        return self.__search_object_for_word_from_map__(word_str=word_str, objects_map=self.glove_wordvec_map)

    def __get_word_vector__(self, node_label_i, is_rnd_init=True):
        start_time = time.time()

        assert node_label_i is not None

        # if not hasattr(self, 'time_to_normalize_wordvec'):
        #     self.time_to_normalize_wordvec = 0

        is_rnd_word_vec_initialized = False

        if (not self.__is_buffer__) or (node_label_i not in self.__wordvec_buffer__):

            word_i_vector = self.get_wordvector_wrapper(node_label_i)

            if is_rnd_init and (word_i_vector is None):
                # random initialization
                if (not hasattr(self, 'npr_state_wordvec_random')) or (self.npr_state_wordvec_random is None):
                    self.npr_state_wordvec_random = npr.RandomState(seed=0)

                word_i_vector = self.npr_state_wordvec_random.randn(self.wordvec_length)
                assert word_i_vector.size == self.wordvec_length

                is_rnd_word_vec_initialized = True

            if word_i_vector is not None:
                l2_norm = word_i_vector.dot(word_i_vector)
                l2_norm = math.sqrt(l2_norm)
                # print l2_norm
                word_i_vector /= l2_norm
                l2_norm = None
                word_i_vector = word_i_vector.astype(np.float32)

            self.__wordvec_buffer__[node_label_i] = word_i_vector
        else:
            word_i_vector = self.__wordvec_buffer__[node_label_i]

        # node_label_i = None

        # if node_label_i is not None:
        #     time_to_load = time.time() - start_time
        #     print '{}: {}, {}'.format(node_label_i, time_to_load, is_rnd_word_vec_initialized)
        #     # self.time_load_wordvecs += time_to_load
        #     # print '{}: {}, {}'.format(node_label_i, time_to_load, self.time_load_wordvecs)
        # else:
        #     print '.',

        return word_i_vector

    def make_tuple_lowercase(self, node_edge_tuple_i):

        node_edge_tuple_i = list(node_edge_tuple_i)

        if len(node_edge_tuple_i) == 2:
            node_edge_tuple_i[1] = node_edge_tuple_i[1].lower()

        node_edge_tuple_i[0] = node_edge_tuple_i[0].lower()

        node_edge_tuple_i = tuple(node_edge_tuple_i)

        return node_edge_tuple_i

    def __compute_kernel_edge_pair__(self, edge_i, edge_j):
        curr_edge_sim = 1

        edge_i = edge_i.lower()
        edge_j = edge_j.lower()

        # print 'edge_i', edge_i
        # print 'edge_j', edge_j

        # assuming that edge labels are already in lower case
        if self.__is_edge_label_sigma_f__ and (edge_i in self.__edge_label_sigma_f_map__):
            curr_edge_sim *= abs(self.__edge_label_sigma_f_map__[edge_i])
            # print 'sigma_f', sigma_f

        if self.__is_edge_label_sigma_f__ and (edge_j in self.__edge_label_sigma_f_map__):
            curr_edge_sim *= abs(self.__edge_label_sigma_f_map__[edge_j])
            # print 'sigma_f', sigma_f

        if curr_edge_sim == 0:
            assert self.__is_edge_label_sigma_f__
            return 0

        curr_edge_pair_tuple = tuple([edge_i, edge_j])

        if edge_i != edge_j:
            if self.__is_edge_label_matrix__ and (curr_edge_pair_tuple in self.__edge_labels_matrix_map__):
                # assert curr_edge_pair_tuple_rev in self.__edge_labels_matrix_map__
                curr_edge_sim *= abs(self.__edge_labels_matrix_map__[curr_edge_pair_tuple])
            else:
                curr_edge_sim *= 0
        else:
            if self.__is_edge_label_matrix__ and (curr_edge_pair_tuple in self.__edge_labels_matrix_map__):
                curr_edge_sim *= abs(self.__edge_labels_matrix_map__[curr_edge_pair_tuple])
            else:
                if self.__is_edge_label_matrix__:
                    curr_edge_sim *= 1
                else:
                    curr_edge_sim *= 1

        if edge_i != edge_j:
            curr_edge_sim = self.proximal_operator(curr_edge_sim, self.__sparsity_proximal_operator_edge_label_kernel__)

        if curr_edge_sim == 0:
            return 0

        assert curr_edge_sim >= 0

        return curr_edge_sim

    def __compute_kernel_node_pair__(self, node_label_i, node_label_j):

        # assert node_label_i is not None
        # assert node_label_j is not None

        k_ij = 1

        if self.__is_node_label_sigma_f__ and (node_label_i in self.__node_label_sigma_f_map__):
            k_ij *= abs(self.__node_label_sigma_f_map__[node_label_i])

        if self.__is_node_label_sigma_f__ and (node_label_j in self.__node_label_sigma_f_map__):
            k_ij *= abs(self.__node_label_sigma_f_map__[node_label_j])

        if k_ij == 0:
            assert self.__is_node_label_sigma_f__
            return 0

        # computing kernel similarity between node labels
        # assert node_edge_tuple_i[0].islower()
        # assert node_edge_tuple_j[0].islower()

        words_tuple = (node_label_i, node_label_j)

        if self.__is_node_label_matrix__ and (words_tuple in self.__node_labels_matrix_map__):
            k_ij *= abs(self.__node_labels_matrix_map__[words_tuple])

            # if node_label_i != node_label_j:
            #     k_ij = self.proximal_operator(k_ij, self.__sparsity_proximal_operator_node_label_kernel__)
        else:
            if self.__is_wordvec_kernel_false__:

                if self.is_wordnet_sim:
                    # start_time_wn = time.time()

                    if node_label_i == node_label_j:
                        k_ij *= 1
                    else:
                        # synset_i = self.get_synset(node_edge_tuple_i)
                        # synset_j = self.get_synset(node_edge_tuple_j)

                        synset_i = self.get_synset(node_label_i, is_word_only=True)
                        synset_j = self.get_synset(node_label_j, is_word_only=True)

                        if (synset_i is not None) and (synset_j is not None):
                            wn_sim = wordnet.wup_similarity(synset_i, synset_j)
                            if wn_sim is None:
                                wn_sim = 0

                            # print 'wn_sim', wn_sim

                            if wn_sim != 0:
                                wn_sim_sparse = self.__sparse_func__(wn_sim, self.__sparse_kernel_threshold__)
                                wn_sim = wn_sim*wn_sim_sparse
                                wn_sim_sparse = None
                            #
                            k_ij *= wn_sim
                            wn_sim = None
                        else:
                            k_ij *= 0

                        # print '{}, {}, {}'.format(synset_i, synset_j, k_ij)
                        synset_i = None
                        synset_j = None

                        # print 'time to compute word net similarity', time.time() - start_time_wn
                else:
                    if node_label_i == node_label_j:
                        k_ij *= 1
                    else:
                        k_ij *= 0
            else:
                if node_label_i != node_label_j:
                    try:
                        node_label_i_wordvec = self.__get_word_vector__(node_label_i)
                        node_label_j_wordvec = self.__get_word_vector__(node_label_j)
                    except Exception as e:
                        print 'Error getting wordvectors'
                        print(e)
                        raise

                    if (node_label_i_wordvec is None) or (node_label_j_wordvec is None):
                        k_ij = 0
                    else:
                        try:
                            k_ij_wordvec = cwkdm.compute_kernel_wordvec(
                                                                node_label_i_wordvec,
                                                                node_label_j_wordvec,
                                                                self.__sparse_kernel_threshold__
                                                            )
                        except Exception as e:
                            print 'Error computing kernel between wordvectors'
                            print(e)
                            raise

                        # print k_ij_wordvec
                        k_ij *= k_ij_wordvec
                        k_ij_wordvec = None

                        # Q_ij = self.get_precomputed_wordvec_cosine_similarity(node_label_i, node_label_j)
                        # k_ij *= Q_ij
                        # # k_ij *= self.__compute_kernel_for_cs__(Q_ij=Q_ij)

        # words_tuple = None
        # assert k_ij >= 0

        return k_ij

    def __compute_kernel_node_edge_pair__(self, node_edge_tuple_i, node_edge_tuple_j):

        assert self.__is_buffer__

        # if node_edge_tuple_i is not None:
        #     node_edge_tuple_i = self.make_tuple_lowercase(node_edge_tuple_i)

        # if node_edge_tuple_j is not None:
        #     node_edge_tuple_j = self.make_tuple_lowercase(node_edge_tuple_j)

        # print '********************'
        # print 'node_edge_tuple_i', node_edge_tuple_i
        # print 'node_edge_tuple_j', node_edge_tuple_j

        # todo: add code to handle the case of reverse tuples
        # if one of the two is reversed and not the other, simply return zero
        # otherwise, bring the tuples to original order and then compute similarity
        assert len(node_edge_tuple_i) == 2
        # if len(node_edge_tuple_i) == 2:
        #     # is_tuple_i_in_order = self.__is_tuple_in_order__(node_edge_tuple_i)
        #     # print 'is_tuple_i_in_order', is_tuple_i_in_order
        # else:
        #     assert len(node_edge_tuple_i) == 1
        #     is_tuple_i_in_order = True

        assert len(node_edge_tuple_j) == 2
        # if len(node_edge_tuple_j) == 2:
        #     is_tuple_j_in_order = self.__is_tuple_in_order__(node_edge_tuple_j)
        #     # print 'is_tuple_j_in_order', is_tuple_j_in_order
        # else:
        #     assert len(node_edge_tuple_j) == 1
        #     is_tuple_j_in_order = True

        # if is_tuple_i_in_order or is_tuple_j_in_order:
        #     if is_tuple_i_in_order and is_tuple_j_in_order:
        #         pass
        #     else:
        #         return 0
        # else:
        #     #  reversing order of tuple
        #     node_edge_tuple_i = self.reverse_tuple(node_edge_tuple_i)
        #     node_edge_tuple_j = self.reverse_tuple(node_edge_tuple_j)

        # is_tuple_i_in_order = None
        # is_tuple_j_in_order = None

        ij_tuple = tuple([node_edge_tuple_i, node_edge_tuple_j])

        # print ij_tuple

        # print 'after preprocessing'
        # print 'node_edge_tuple_i', node_edge_tuple_i
        # print 'node_edge_tuple_j', node_edge_tuple_j

        if ij_tuple not in self.__kernel_buffer_node_edge_tuple__:

            # self.count_kernel_buffer_node_edge_tuple += 1
            # print 'count_kernel_buffer_node_edge_tuple', self.count_kernel_buffer_node_edge_tuple

            k_ij = 1

            if (len(node_edge_tuple_i) == 2) and (len(node_edge_tuple_j) == 2):
                if self.is_edge_label_sim:
                    edge_i = node_edge_tuple_i[1]
                    edge_j = node_edge_tuple_j[1]

                    edge_tuple = tuple([edge_i, edge_j])

                    if edge_tuple not in self.__kernel_buffer_edge_tuple__:
                        curr_edge_sim = self.__compute_kernel_edge_pair__(edge_i, edge_j)
                        self.__kernel_buffer_edge_tuple__[edge_tuple] = curr_edge_sim
                        self.__kernel_buffer_edge_tuple__[tuple(reversed(edge_tuple))] = curr_edge_sim
                    else:
                        # print '.',
                        curr_edge_sim = self.__kernel_buffer_edge_tuple__[edge_tuple]

                    edge_tuple = None

                    if curr_edge_sim == 0:
                        self.__kernel_buffer_node_edge_tuple__[ij_tuple] = 0
                        self.__kernel_buffer_node_edge_tuple__[tuple(reversed(ij_tuple))] = 0
                        return 0

                    assert k_ij >= 0

                    k_ij *= curr_edge_sim
                    curr_edge_sim = None
                else:
                    pass
            elif len(node_edge_tuple_i) != len(node_edge_tuple_j):
                self.__kernel_buffer_node_edge_tuple__[ij_tuple] = 0
                self.__kernel_buffer_node_edge_tuple__[tuple(reversed(ij_tuple))] = 0
                return 0
            elif (len(node_edge_tuple_i) == 1) and (len(node_edge_tuple_j) == 1):
                edge_i = None
                edge_j = None
            else:
                raise AssertionError

            assert k_ij >= 0

            node_tuple = tuple([node_edge_tuple_i[0], node_edge_tuple_j[0]])

            if node_tuple not in self.__kernel_buffer_node_tuple__:
                curr_node_sim = self.__compute_kernel_node_pair__(node_edge_tuple_i[0], node_edge_tuple_j[0])
                self.__kernel_buffer_node_tuple__[node_tuple] = curr_node_sim
                self.__kernel_buffer_node_tuple__[tuple(reversed(node_tuple))] = curr_node_sim
            else:
                # print '.',
                curr_node_sim = self.__kernel_buffer_node_tuple__[node_tuple]

            node_tuple = None

            if curr_node_sim == 0:
                self.__kernel_buffer_node_edge_tuple__[ij_tuple] = 0
                self.__kernel_buffer_node_edge_tuple__[tuple(reversed(ij_tuple))] = 0
                return 0

            k_ij *= curr_node_sim

            self.__kernel_buffer_node_edge_tuple__[ij_tuple] = k_ij
            self.__kernel_buffer_node_edge_tuple__[tuple(reversed(ij_tuple))] = k_ij
            return k_ij
        else:
            # print '+',
            # self.count_kernel_buffer_node_edge_tuple_reused += 1
            # print 'count_kernel_buffer_node_edge_tuple_reused', self.count_kernel_buffer_node_edge_tuple_reused

            return self.__kernel_buffer_node_edge_tuple__[ij_tuple]

    def __compute_kernel_matrix_parallel__(self, amr_graphs1, amr_graphs2, is_sparse, is_normalize):
        assert is_sparse

        start_time = time.time()

        num_cores = self.num_cores
        print 'num_cores', num_cores

        kernel_matrix_queue = [mp.Queue() for d in range(num_cores)]
        n1 = amr_graphs1.size
        n2 = amr_graphs2.size

        K = ss.dok_matrix((n1, n2))
        idx_range_parallel = pc.uniform_distribute_tasks_across_cores(n1, num_cores)
        processes = [
            mp.Process(
                target=self.__eval_kernel_parallel_wrapper__,
                args=(
                    amr_graphs1[idx_range_parallel[currCore]],
                    amr_graphs2,
                    is_sparse,
                    is_normalize,
                    kernel_matrix_queue[currCore]
                )
            ) for currCore in range(num_cores)
        ]

        #start processes
        for process in processes:
            process.start()

        for currCore in range(num_cores):
            if self.__is_coarse_debug__:
                print('waiting for results from core ', currCore)
            # todo: this should not work, replace with mesh index
            result = kernel_matrix_queue[currCore].get()
            if isinstance(result, BaseException) or isinstance(result, OSError): #it means that subprocess has an error
                print 'a child processed has thrown an exception. raising the exception in the parent process to terminate the program'
                print 'one of the child processes failed, so killing all child processes'
                #kill all subprocesses
                for process in processes:
                    if process.is_alive():
                        process.terminate() #assuming that the child process do not have its own children (those granchildren would be orphaned with terminate() if any)
                print 'killed all child processes'
                raise result
            else:
                result = result.todok()
                K[idx_range_parallel[currCore], :] = result
            #
            if self.__is_coarse_debug__:
                print('got results from core ', currCore)

        kernel_matrix_queue = None
        #wait for processes to complete
        for process in processes:
            process.join()

        K = K.tocsr()

        print 'Time to compute the matrix of shape {} is {}'.format(K.shape, time.time()-start_time)

        return K

    def __eval_kernel_parallel_wrapper__(self, amr_graphs1, amr_graphs2, is_sparse, is_normalize, kernel_matrix_queue):
        try:
            K = self.__compute_kernel_matrix__(amr_graphs1, amr_graphs2, is_sparse, is_normalize)
            assert K is not None
            assert K.shape[0] == amr_graphs1.shape[0]
            assert K.shape[1] == amr_graphs2.shape[0]
            #
            kernel_matrix_queue.put(K)
        except BaseException as e:
            print 'error in the subprocess (base exception)'
            print e
            kernel_matrix_queue.put(e)
        except OSError as ee:
            print 'error in the subprocess (os error)'
            print ee
            kernel_matrix_queue.put(ee)
        except:
            print 'error in the subprocess (any exception)'
            print e
            kernel_matrix_queue.put(e)

    def __compute_kernel_matrix__(self, arr_text_features_data1, arr_text_features_data2, is_sparse, is_normalize):

        self.lamb_sqr = self.__lamb__**2

        n1 = arr_text_features_data1.size
        n2 = arr_text_features_data2.size

        start_time = time.time()

        if not is_sparse:
            K = -float('Inf')*np.ones((n1, n2))
        else:
            K = ss.dok_matrix((n1, n2))

        is_error = False
        num_kernel_errors = 0

        if is_normalize:
            self_k_i_sqrt = -float('Inf')*np.ones(n1)
            self_k_j_sqrt = -float('Inf')*np.ones(n2)

        # note: do not exploit symmetry of kernel matrix since the this function may be called from a parallel computing
        #  module for computing sub-matrices

        for i in range(n1):

            # print '*****************************'
            # print arr_text_features_data1[i]['path_tuple']

            for j in range(n2):

                # print '*****************************'

                try:
                    start_time_local = time.time()
                    list_i = arr_text_features_data1[i]['path_tuple']
                    if self.__is_debug__:
                        print list_i

                    list_j = arr_text_features_data2[j]['path_tuple']
                    if self.__is_debug__:
                        print list_j

                    if list_i is None or list_j is None:
                        Kij = 0
                    else:
                        Kij = self.compute_kernel(
                            list_i,
                            list_j,
                        )
                    if self.__is_debug__:
                        print 'Kij ', Kij

                    if is_normalize and (Kij != 0):
                        # assuming that computations are saved in a global map so that we don't have to bother about same calls from here
                        if self_k_i_sqrt[i] == -float('Inf'):
                            Kii = self.compute_kernel(
                                list_i,
                                list_i
                            )
                            self_k_i_sqrt[i] = math.sqrt(Kii)
                        else:
                            Kii = self_k_i_sqrt[i]
                        if self.__is_debug__:
                            print 'Kii ', Kii
                        #
                        if self_k_j_sqrt[j] == -float('Inf'):
                            Kjj = self.compute_kernel(
                                list_j,
                                list_j
                            )
                            self_k_j_sqrt[j] = math.sqrt(Kjj)
                        else:
                            Kjj = self_k_j_sqrt[j]
                        if self.__is_debug__:
                            print 'Kjj ', Kjj

                        # assert Kii != 0
                        # assert Kjj != 0

                        #normalized K
                        if (Kii == 0) or (Kjj == 0):
                            if self.__is_debug__:
                                print 'warning: Kii or Kjj is zero.'
                            K[i, j] = 0
                        else:
                            K[i, j] = Kij/(Kii*Kjj)

                        # #truncation for memory savings and computations
                        if -0.1 < K[i, j] <= 0:
                            K[i, j] = 0
                        # elif 1.1 > K[i, j] >= 1:
                        #     K[i, j] = 1
                        #
                        # if self.__is_debug__:
                        #     print 'normalized K[i, j] is ', K[i, j]
                        # if not (0 <= K[i,j] <= 1):
                        #
                        if K[i, j] < 0:
                            print '******************************************'
                            print 'unexpected value of K[i,j] ', K[i, j]
                            print 'Kij is ', Kij
                            print 'Kii is ', Kii
                            print 'Kjj is ', Kjj
                            print arr_text_features_data1[i]['path_tuple']
                            print arr_text_features_data2[j]['path_tuple']
                            print '******************************************'
                            raise AssertionError
                    else:
                        K[i, j] = Kij
                    #
                    if self.__is_debug__:
                        # if K[i, j] > 0:
                        print 'K[i, j] is ', K[i, j]
                        # if self.__is_debug__:
                        print 'time to computer current kernel was ', time.time()-start_time_local
                except:
                        print 'error computing kernel between '
                        print arr_text_features_data1[i]['path_tuple']
                        print arr_text_features_data2[j]['path_tuple']
                        is_error = True
                        num_kernel_errors += 1
                        raise
                        # if num_kernel_errors > 100:
                        #     raise
                if self.__is_debug__:
                    print 'K[i, j] ', K[i, j]
        if is_error:
            print 'Number of kernel computation errors ', num_kernel_errors
            raise AssertionError
        if self.__is_debug__:
            print 'Kernel Matrix is ', K

        if is_sparse:
            K = K.tocsr()

        if self.__is_coarse_debug__:
            print 'Time to compute the kernel matrix is ', time.time()-start_time

        return K

    def __compute_kernel_normalization__(self, arr_text_features_data):
        print 'arr_text_features_data.shape', arr_text_features_data.shape
        n = arr_text_features_data.shape[0]
        #
        start_time = time.time()
        #
        k = -float('Inf')*np.ones(n)
        #
        for i in range(n):
            list_i = arr_text_features_data[i]['path_tuple']
            # print 'list_i', list_i
            #
            if list_i is not None:
                list_i = self.__get_embedding_fr_tuple__(list_i)
            #
            if (list_i is None) or (not list_i):
                curr_norm = 1
            else:
                curr_norm \
                    = self.compute_kernel(
                    list_i,
                    list_i,
                )
                if curr_norm == 0:
                    curr_norm = 1
            #
            k[i] = curr_norm
            # print 'k[i]', k[i]
        # print 'Time to compute the kernel matrix is ', time.time()-start_time
        return k

    def compute_kernel(self, word_vec_list1, word_vec_list2, is_cython=True):
        # adaptation of code from link below for academic purposes.
        # https://github.com/arne-cl/alt-mulig/blob/master/kernel-methods/string-subsequence-kernel.ipynb

        if (word_vec_list1 is None) or (word_vec_list2 is None):
            return 0

        if (not word_vec_list1) or (not word_vec_list2):
            return 0

        # print '..............................'
        # print 'word_vec_list1', len(word_vec_list1)
        # print 'word_vec_list2', len(word_vec_list2)
        # if max(len(word_vec_list1), len(word_vec_list2)) > 200:
        #     print 'wohoo!'

        num_terms_1 = len(word_vec_list1)
        num_terms_2 = len(word_vec_list2)

        lamb_sqr = self.lamb_sqr
        lamb = self.__lamb__

        # start_time = time.time()

        matches = np.zeros(dtype=np.bool, shape=(num_terms_1, num_terms_2))
        dps = np.zeros((num_terms_1, num_terms_2), dtype=np.float32)
        for curr_idx_1 in xrange(num_terms_1):
            term1 = word_vec_list1[curr_idx_1]
            for curr_idx_2 in xrange(num_terms_2):
                term2 = word_vec_list2[curr_idx_2]
                #
                try:
                    curr_k = self.__compute_kernel_node_edge_pair__(term1, term2)
                except Exception as e:
                    print 'error computing basic kernel similarity.'
                    print(e)
                    raise
                #
                if curr_k != 0:
                    matches[curr_idx_1, curr_idx_2] = True
                    # dps[curr_idx_1, curr_idx_2] = lamb_sqr*curr_k
                    dps[curr_idx_1, curr_idx_2] = lamb_sqr

        # print dps

        # print 'Time to compute word similarities', time.time()-start_time

        start_time = time.time()
        dp = np.zeros((num_terms_1+1, num_terms_2+1), dtype=np.float32)
        k = np.zeros(self.__p__+1, dtype=np.float32)

        if is_cython:
            matches = matches.astype(np.int32)
            # p_weights = self.__p_weights__.astype(np.float32)
            weighted_sum_k = skc.compute_subsequence_kernel_from_word_similarities(
                num_terms_1=num_terms_1,
                num_terms_2=num_terms_2,
                p=self.__p__,
                lamb_sqr=lamb_sqr,
                lamb=lamb,
                dps=dps,
                dp=dp,
                k=k,
                matches=matches,
                p_weights=self.__p_weights__,
            )
            # p_weights=None
        else:
            k[1] = lamb*dps.sum()
            for l in xrange(2, self.__p__+1):
                for i in xrange(num_terms_1):
                    for j in xrange(num_terms_2):
                        dp[i+1, j+1] = dps[i, j] + lamb * dp[i, j+1] + lamb * dp[i+1, j] - lamb_sqr * dp[i, j]
                        if matches[i, j]:
                            dps[i, j] = lamb_sqr * dp[i, j]
                            k[l] = k[l] + dps[i, j]

            weighted_sum_k_expr = k*self.__p_weights__
            weighted_sum_k = weighted_sum_k_expr.sum()
            weighted_sum_k_expr = None

        # print 'Time to compute convolution kernel ', time.time()-start_time

        return weighted_sum_k
