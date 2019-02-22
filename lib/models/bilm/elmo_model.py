from __future__ import absolute_import

import json
import numpy as np
import tensorflow as tf
from lib.models.bilm.data import ElmoBatcher
from lib.models.bilm.model import BidirectionalLanguageModel
from lib.models.bilm.elmo import weight_layers


class ElmoLSTMEncoder(object):
  # def __init__(self, text_batch, e1_dist_batch, e2_dist_batch, seq_len_batch, lstm_dim, embed_dim, position_dim,
  #              token_dim, bidirectional, peephole, max_pool, word_dropout_keep, lstm_dropout_keep,
  #              final_dropout_keep, FLAGS, entity_index=100, filterwidth=3, pos_encode_batch=None):
  def __init__(self, dataset):
    # todo don't hardcode these
    # options_file = '/iesl/canvas/strubell/Parser/elmo_model/elmo_2x4096_512_2048cnn_2xhighway_options.json'
    # weight_file = '/iesl/canvas/strubell/Parser/elmo_model/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'
    options_file = 'elmo_model/elmo_2x4096_512_2048cnn_2xhighway_options.json'
    weight_file = 'elmo_model/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'
    vocab_file = dataset.word_file
    with open(options_file, 'r') as fin:
      options = json.load(fin)
    max_word_length = options['char_cnn']['max_characters_per_token']

    # TODO: make sure elmo frozen
    # vocab = UnicodeCharsVocabulary(vocab_file, max_word_length)
    self.elmo_batcher = ElmoBatcher(vocab_file, max_word_length)

    default_elmo = np.ones((1, 3, max_word_length), dtype=np.int32)
    self.elmo_ids_placeholder = tf.placeholder_with_default(default_elmo,
                                                            shape=(None, None, max_word_length),
                                                            name='elmo_characters'
                                                            )
    # todo max batch size set wrong
    self.elmo = BidirectionalLanguageModel(options_file, weight_file, max_batch_size=dataset.max_batch_size())
    self.elmo_ops = self.elmo(self.elmo_ids_placeholder)

    # super(ElmoLSTMEncoder, self).__init__(text_batch, e1_dist_batch, e2_dist_batch, seq_len_batch, lstm_dim,
    #                                       embed_dim, position_dim, token_dim, bidirectional, peephole, max_pool,
    #                                       word_dropout_keep, lstm_dropout_keep, final_dropout_keep, FLAGS,
    #                                       entity_index,
    #                                       pos_encode_batch=pos_encode_batch)
    # super(ElmoLSTMEncoder, self).__init__(*args, **kwargs)
    self.model_type = 'elmo'

    # self.vocabs = dataset.vocabs
    # self.dataset = dataset

  def get_feed_dict(self, feed_dict, sents):
    # e1, e2, ep, rel, tokens, e1_dist, e2_dist, seq_len = batch
    # token_map = string_int_maps['token_id_str_map']
    # remove pad tokens
    # todo 0 or 1?
    # tokens_batch = feed_dict[self.dataset.inputs][:, :, 1]
    # print("feed: ", feed_dict[self.dataset.inputs].shape, feed_dict[self.dataset.inputs][:, 0])
    # str_tokens = [[self.vocabs[0][t] for t in sentence] for sentence in tokens_batch]
    # print("sents", sents)
    # print("str tokens: ", str_tokens)
    # map text to sentences
    char_ids = self.elmo_batcher.batch_sentences(sents)
    feed_dict[self.elmo_ids_placeholder] = char_ids
    return feed_dict

  def embed_text(self):
    # learned weighted average over the elmo layers
    elmo_embeddings = weight_layers('elmo_input', self.elmo_ops, l2_coef=0.0)['weighted_op']
    # elmo_embeddings = tf.nn.dropout(elmo_embeddings, self.word_dropout_keep)

    # e1_pos, e2_pos = self.get_position_embeddings(e1_dist_batch, e2_dist_batch, position_embeddings)
    # token_embeds = tf.concat(axis=2, values=[elmo_embeddings, e1_pos, e2_pos])
    # token_embeds = self.get_token_embeddings(token_embeddings, position_embeddings)
    # token_embeds = tf.Print(token_embeds, [tf.shape(elmo_embeddings)])

    # # project elmo embeddings down to input for relation encoder
    # dim = self.lstm_dim - (2*self.position_dim)
    # params = {"inputs": token_embeds, "filters": dim, "kernel_size": 1,
    #           "activation": tf.nn.leaky_relu, "use_bias": True, "padding": "same", "dilation_rate": 1}
    # token_embeds = tf.layers.conv1d(**params)

    # self.sentence_embeddings = self.embed_text_from_tokens(
    #   token_embeds, ep_embeddings, e1_dist_batch, e2_dist_batch, seq_len_batch, reuse=reuse)
    return elmo_embeddings