from auto_cen.pusion.core.decision_templates_combiner import *


class DempsterShaferCombiner(TrainableCombiner):
    """
    The :class:`DempsterShaferCombiner` (DS) fuses decision outputs by means of the Dempster Shafer evidence theory
    referenced by Polikar :footcite:`polikar2006ensemble` and Ghosh et al. :footcite:`ghosh2011evaluation`.
    DS involves computing the `proximity` and `belief` values per classifier and class, depending on a sample.
    Then, the total class support is calculated using the Dempster's rule as the product of belief values across all
    classifiers to each class, respectively. The class with the highest product is considered as a fused decision.
    DS shares the same training procedure with the :class:`DecisionTemplatesCombiner`.

    :param similarity: The similarity metric used to compute the distance.
                        Euclidean, Manhattan, Cosine and Symmetric Difference are supported.

    .. footbibliography::
    """

    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.REDUNDANT),
        (Problem.MULTI_CLASS, AssignmentType.CONTINUOUS, CoverageType.REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CRISP, CoverageType.REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CONTINUOUS, CoverageType.REDUNDANT),
    ]

    SHORT_NAME = 'DS'

    def __init__(self, similarity: str = 'euclidean'):
        TrainableCombiner.__init__(self)
        self.similarity = similarity

        self.decision_templates = None
        self.distinct_class_assignments = None

    def train(self, decision_tensor: np.array, true_assignments: np.array):
        """
        Train the Dempster Shafer Combiner model by precalculating decision templates from given decision outputs and
        true class assignments. Both continuous and crisp classification outputs are supported. This procedure involves
        calculations mean decision profiles (decision templates) for each true class assignment.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of either crisp or continuous decision outputs by different classifiers per sample.

        :param true_assignments: `numpy.array` of shape `(n_samples, n_classes)`.
                Matrix of either crisp or continuous class assignments which are considered true for each sample during
                the training procedure.
        """
        dt_combiner = DecisionTemplatesCombiner(self.similarity)
        dt_combiner.train(decision_tensor, true_assignments)
        self.decision_templates = dt_combiner.get_decision_templates()
        self.distinct_class_assignments = dt_combiner.get_distinct_class_assignments()

    def combine(self, decision_tensor: np.array) -> np.array:
        """
        Combine decision outputs by using the Dempster Shafer method.
        Both continuous and crisp classification outputs are supported. Combining requires a trained
        :class:`DempsterShaferCombiner`.
        This procedure involves computing the proximity, the belief values, and the total class support using the
        Dempster's rule.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of either crisp or continuous decision outputs by different classifiers per sample.

        :return: A matrix (`numpy.array`) of either crisp or continuous class assignments which represents fused
                decisions obtained by the maximum class support. Axis 0 represents samples and axis 1 the class
                assignments which are aligned with axis 2 in ``decision_tensor`` input tensor.
        """
        decision_profiles = decision_tensor_to_decision_profiles(decision_tensor)
        fused_decisions = np.zeros_like(decision_tensor[0])

        for i in range(len(decision_profiles)):
            dp = decision_profiles[i]
            n_label = len(self.decision_templates)
            n_classifiers = len(decision_tensor)

            # Compute support for each label (Dempster's rule)
            mu = self._support(dp, n_label, n_classifiers)
            fused_decisions[i] = self.distinct_class_assignments[np.argmax(mu)]
        return fused_decisions

    def _proximity(self, dp: np.array, n_label: int, n_classifiers: int) -> np.array:
        """
        Compute the proximities of each classifier and class to the prediction.

        :param dp: A row of the decision profile, i.e. the predictions
        :param n_label: The number of classes
        :param n_classifiers: The number of classifiers

        :return: A matrix of dimensions (n_classes, n_classifiers) holding the proximity values for
                 each classifier-class pair

        """
        prox = np.empty((n_label, n_classifiers))  # Phi_{j,k}
        for j in range(n_label):
            dt = self.decision_templates[j]
            for k in range(n_classifiers):
                d = 0.0
                # Compute the normalisation
                for j_ in range(n_label):
                    d += (1 + np.linalg.norm(self.decision_templates[j_][k] - dp[k])) ** (-1)

                prox[j, k] = (1 + np.linalg.norm(dt[k] - dp[k])) ** (-1) / d
        return prox

    def _belief(self, dp: np.array, n_label: int, n_classifiers: int) -> np.array:
        """
        Computes the belief of each classifier and class to the prediction


        :param dp: A row of the decision profile, i.e. the predictions
        :param n_label: The number of classes
        :param n_classifiers: The number of classifiers

        :return: A matrix of dimensions (n_classes, n_classifiers) holding the belief values for
                 each classifier-class pair
        """
        prox = self._proximity(dp, n_label, n_classifiers)
        bel = np.empty((n_label, n_classifiers))  # bel_{j,k}
        for j in range(n_label):
            for k in range(n_classifiers):
                prod = 1.0
                for j_ in range(n_label):
                    if j_ != j:
                        prod = prod * (1 - prox[j_, k])

                bel[j, k] = prox[j, k] * prod / (1 - prox[j, k] * (1 - prod))
        return bel

    def _support(self, dp: np.array, n_label: int, n_classifiers: int) -> np.array:
        """
        Computes the dempster-shafer support for each class.


        :param dp: A row of the decision profile, i.e. the predictions
        :param n_label: The number of classes
        :param n_classifiers: The number of classifiers

        :return: A vector of dimension (n_classes) holding the belief values for
                 each class.
        """

        bel = self._belief(dp, n_label, n_classifiers)

        mu = np.zeros(n_label)
        for j in range(n_label):
            prod = 1.0
            for k in range(n_classifiers):
                prod = prod * bel[j, k]
            mu[j] = prod

        # normalization
        return mu / np.sum(mu)
