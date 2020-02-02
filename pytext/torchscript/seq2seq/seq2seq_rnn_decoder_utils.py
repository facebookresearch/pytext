#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


def get_src_length(models, decoder_ip):
    for i, model in enumerate(models):
        return model.get_src_length(decoder_ip[i])


def prepare_decoder_ips(models, decoder_ip, model_state_outputs, prev_hypos):

    decoder_ips = []

    for i, (model, states) in enumerate(zip(models, model_state_outputs)):
        src_tokens, src_lengths = model.get_src_tokens_lengths(decoder_ip)
        encoder_rep = model.get_encoder_rep(decoder_ip[i])

        prev_hiddens = states[0]
        prev_cells = states[1]
        attention = states[2]

        prev_hiddens_for_next = []
        for hidden in prev_hiddens:
            prev_hiddens_for_next.append(hidden.index_select(dim=0, index=prev_hypos))

        prev_cells_for_next = []
        for cell in prev_cells:
            prev_cells_for_next.append(cell.index_select(dim=0, index=prev_hypos))

        attention_for_next = attention.index_select(dim=0, index=prev_hypos)

        decoder_ips.append(
            (
                encoder_rep,
                tuple(prev_hiddens_for_next),
                tuple(prev_cells_for_next),
                attention_for_next,
                src_tokens,
                src_lengths,
            )
        )

    return tuple(decoder_ips)
