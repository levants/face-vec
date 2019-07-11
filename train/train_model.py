"""
Created on Jul 11, 2019

Utility module for training face representation

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets.data_utils import get_input_tensors


# For current batch --> get tensors --> forward --> find pos, neg dist -->
# check if: neg_dist< pos_dist + margin --> true: include all true pairs --> forward to get loss --> find grad/step
def mine_semi_hard(forward, cur_batch, conv_net, dense_net, margin):
    # For this database of imags margin used was around 0.6
    # Probably using dynamic margins should work even better

    conv_net.eval()
    dense_net.eval()

    batch_imgs_anc, batch_imgs_pos, batch_imgs_neg = get_input_tensors(cur_batch)

    anc_fc_out = forward(batch_imgs_anc, conv_net, dense_net)

    pos_fc_out = forward(batch_imgs_pos, conv_net, dense_net)

    neg_fc_out = forward(batch_imgs_neg, conv_net, dense_net)

    pos_dist = (anc_fc_out - pos_fc_out).norm(2, dim=1)

    neg_dist = (anc_fc_out - neg_fc_out).norm(2, dim=1)  # - margin

    semi_hard_idx = (neg_dist < pos_dist + margin).nonzero()

    semi_hard_cur_batch = [cur_batch[i] for i in semi_hard_idx]  # weeds out easy ones

    conv_net.train()
    dense_net.train()

    return semi_hard_cur_batch
