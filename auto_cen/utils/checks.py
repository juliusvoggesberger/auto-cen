"""
Module containing checks that compare an algorithm specification (combiner/classifier) to a problem specification or
that check an algorithms specification for correctness.
"""

from auto_cen.constants import BINARY, CONTINUOUS_OUT, LABELS


def classifier_satisfies_specification(classifier_spec: dict, specification: dict) -> bool:
    """
    Check if a classifier conforms to a problem specification.

    :param classifier_spec: Specification of the classifier.
    :param specification:  Specification of the problem.
    :return: If it conforms: True, Else: False.
    """

    # If a predefined specification was given, then the output is already defined.
    if 'output' in specification and not (
            set(specification['output']).issubset(classifier_spec['output']) or set(
        classifier_spec['output'])
            .issubset(specification['output'])):
        return False
    else:
        return (set(specification['problem']).issubset(classifier_spec['problem']) or
                specification['problem'][0] == BINARY)
        # and (specification['input'] == classifier_spec['input'] or (
        # specification['input'] == NUMERICAL and classifier_spec['input'] == MIXED) or (
        #                      specification['input'] == CATEGORICAL and classifier_spec['input'] == MIXED))


def combiner_satisfies_specification(combiner_spec: dict, specification: dict) -> bool:
    """
    Check if a combiner conforms to a problem specification.

    :param combiner_spec: Specification of the combiner.
    :param specification:  Specification of the problem.
    :return: If it conforms: True, Else: False.
    """
    # Only select combiners which:
    # 1. Can solve the given problem -> If Problem is Binary, every combiner can solve that
    # 2. Can take the output as input -> works as long as the output is subset of input or otherwise
    return (set(specification['problem']).issubset(combiner_spec['problem']) or
            specification['problem'][0] == BINARY) \
           and (set(specification['output']).issubset(combiner_spec['input']) or set(
        combiner_spec['input'])
                .issubset(specification['output']))

