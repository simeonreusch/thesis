# from estimate_explosion_time.shared import get_custom_logger, main_logger_name
# import logging
#
# logger = get_custom_logger(main_logger_name)
# logger.setLevel(logging.DEBUG)

import json
import argparse
import os
import numpy as np


class ReduceError(Exception):
    def __init__(self, msg):
        self.msg = msg


def read_mosfit_output(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())
        if 'name' not in data:
            data = data[list(data.keys())[0]]
    return data


def get_explosion_time_posterior(mosfit_output_data):

    model = mosfit_output_data['models'][0]
    # photometry = data['photometry']
    max_score = 0
    max_realization_ind = None
    texp = []
    tref = None

    for i, realization in enumerate(model['realizations']):

        this_score = float(realization['score'])
        if this_score > max_score:
            max_score = this_score
            max_realization_ind = i

        this_tref = realization['parameters']['reference_texplosion']['value']
        if i == 0:
            tref = this_tref
        elif tref != this_tref:
            raise ReduceError('different reference times!')

        texp.append(float(realization['parameters']['texplosion']['value']))

    return texp, tref, max_realization_ind


def reduce_mosfit_output_photometry(mosfit_output_data, confidence_level):

    photometry = mosfit_output_data['photometry']

    real_data = [x for x in photometry if 'band' in x and 'magnitude' in x and 'realization' not in x]
    data_from_fits = [x for x in photometry if 'band' in x and 'realization' in x]

    ordered_data_from_fits = {}

    for x in data_from_fits:

        if not x['band'] in ordered_data_from_fits.keys():
            ordered_data_from_fits[x['band']] = {}

        if not x['time'] in ordered_data_from_fits[x['band']].keys():
            ordered_data_from_fits[x['band']][x['time']] = []

        ordered_data_from_fits[x['band']][x['time']].append(float(x['magnitude']))

    new_data_from_fits = []
    for (band, time_dict) in ordered_data_from_fits.items():
        for (time, magnitudes) in time_dict.items():
            ci = np.quantile(magnitudes, [0.5 - confidence_level / 2, 0.5 + confidence_level / 2])
            new_data_from_fits.append({
                'band': band,
                'time': time,
                'u_time': 'MJD',
                'realization': 'all',
                'confidence_level': str(confidence_level),
                'confidence_interval_upper': str(ci[0]),
                'confidence_interval_lower': str(ci[1])
            })

    return real_data, new_data_from_fits


def make_reduced_output(data, confidence_level=0.9):

    if isinstance(data, str) and os.path.isfile(data):
        data = read_mosfit_output(data)

    texp, tref, max_realization_ind = get_explosion_time_posterior(data)

    texp_q = np.quantile(texp, [0.5 - confidence_level / 2, 0.5, 0.5 + confidence_level / 2])

    dict_to_keep = [{
        'parameters': {
            'texplosion': {
                'confidence_interval_upper': str(texp_q[2]),
                'confidence_interval_lower': str(texp_q[0]),
                'median': str(texp_q[1]),
                'gaussian_error': str(np.std(texp_q))
            },
            'reference_texplosion': str(tref)
        }
    }]

    model = data['models'][0]
    model['best_realization'] = model['realizations'][max_realization_ind]
    model['realizations'] = dict_to_keep
    data['models'][0] = model

    real_data, new_data_from_fits = reduce_mosfit_output_photometry(data, confidence_level=confidence_level)
    data['photometry'] = real_data + new_data_from_fits

    return data


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='only keep sepcified parameters in realizations')
    parser.add_argument('file', type=str, help='path to json file')
    parser.add_argument('--confidence_level', type=float,
                        help='confidence level', default=0.9)
    args = parser.parse_args()

    new_data = make_reduced_output(args.file, args.confidence_level)

    with open(args.file, 'w') as f:
        json.dump(new_data, f, indent=4, sort_keys=True)
