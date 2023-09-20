# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from collections import OrderedDict

from tensorflow.keras import Model

# from .utils import global_mutable_counting


_logger = logging.getLogger(__name__)

_counter = 0

def global_mutable_counting():
    global _counter
    _counter += 1
    return _counter

class Mutable(Model):
    def __init__(self, key=None):
        super().__init__()
        if key is None:
            self._key = '{}_{}'.format(type(self).__name__, global_mutable_counting())
        elif isinstance(key, str):
            self._key = key
        else:
            self._key = str(key)
            _logger.warning('Key "%s" is not string, converted to string.', key)
        self.init_hook = None
        self.forward_hook = None

    def __deepcopy__(self, memodict=None):
        raise NotImplementedError("Deep copy doesn't work for mutables.")

    def set_mutator(self, mutator):
        if hasattr(self, 'mutator'):
            raise RuntimeError('`set_mutator is called more than once. '
                               'Did you parse the search space multiple times? '
                               'Or did you apply multiple fixed architectures?')
        self.mutator = mutator

    def call(self, *inputs):
        raise NotImplementedError('Method `call` of Mutable must be overridden')

    def build(self, input_shape):
        self._check_built()

    @property
    def key(self):
        return self._key

    @property
    def name(self):
        return self._name if hasattr(self, '_name') else self._key

    @name.setter
    def name(self, name):
        self._name = name

    def _check_built(self):
        if not hasattr(self, 'mutator'):
            raise ValueError(
                "Mutator not set for {}. You might have forgotten to initialize and apply your mutator. "
                "Or did you initialize a mutable on the fly in forward pass? Move to `__init__` "
                "so that trainer can locate all your mutables. See NNI docs for more details.".format(self))

    def __repr__(self):
        return '{} ({})'.format(self.name, self.key)


class MutableScope(Mutable):
    def __call__(self, *args, **kwargs):
        try:
            self.mutator.enter_mutable_scope(self)
            return super().__call__(*args, **kwargs)
        finally:
            self.mutator.exit_mutable_scope(self)


class LayerChoice(Mutable):
    def __init__(self, op_candidates, reduction='sum', return_mask=False, key=None):
        super().__init__(key=key)
        self.names = []
        if isinstance(op_candidates, OrderedDict):
            for name in op_candidates:
                assert name not in ["length", "reduction", "return_mask", "_key", "key", "names"], \
                    "Please don't use a reserved name '{}' for your module.".format(name)
                self.names.append(name)
        elif isinstance(op_candidates, list):
            for i, _ in enumerate(op_candidates):
                self.names.append(str(i))
        else:
            raise TypeError("Unsupported op_candidates type: {}".format(type(op_candidates)))

        self.length = len(op_candidates)
        self.choices = op_candidates
        self.reduction = reduction
        self.return_mask = return_mask

    def call(self, *inputs):
        out, mask = self.mutator.on_forward_layer_choice(self, *inputs)
        if self.return_mask:
            return out, mask
        return out

    def build(self, input_shape):
        self._check_built()
        for op in self.choices:
            op.build(input_shape)

    def __len__(self):
        return len(self.choices)


class InputChoice(Mutable):
    NO_KEY = ''

    def __init__(self, n_candidates=None, choose_from=None, n_chosen=None, reduction='sum', return_mask=False, key=None):
        super().__init__(key=key)
        assert n_candidates is not None or choose_from is not None, \
                'At least one of `n_candidates` and `choose_from` must be not None.'
        if choose_from is not None and n_candidates is None:
            n_candidates = len(choose_from)
        elif choose_from is None and n_candidates is not None:
            choose_from = [self.NO_KEY] * n_candidates
        assert n_candidates == len(choose_from), 'Number of candidates must be equal to the length of `choose_from`.'
        assert n_candidates > 0, 'Number of candidates must be greater than 0.'
        assert n_chosen is None or 0 <= n_chosen <= n_candidates, \
                'Expected selected number must be None or no more than number of candidates.'

        self.n_candidates = n_candidates
        self.choose_from = choose_from.copy()
        self.n_chosen = n_chosen
        self.reduction = reduction
        self.return_mask = return_mask

    def call(self, optional_inputs):
        optional_input_list = optional_inputs
        if isinstance(optional_inputs, dict):
            optional_input_list = [optional_inputs[tag] for tag in self.choose_from]
        assert isinstance(optional_input_list, list), \
                'Optional input list must be a list, not a {}.'.format(type(optional_input_list))
        assert len(optional_inputs) == self.n_candidates, \
                'Length of the input list must be equal to number of candidates.'
        out, mask = self.mutator.on_forward_input_choice(self, optional_input_list)
        if self.return_mask:
            return out, mask
        return out
    
class LabeledMutable(Mutable):
    """:class:`Mutable` with a label. This should be the super-class of most mutables.
    The labels are widely used in simplified result, as well as samples.
    Usually a mutable must be firstly converted into one or several :class:`LabeledMutable`,
    before strategy can recognize and process it.

    When two mutables have the same label, they semantically share the same choice.
    That means, the choices of the two mutables will be shared.
    The labels can be either auto-generated, or provided by the user.

    Being a :class:`LabeledMutable` doesn't necessarily mean that it is a leaf mutable.
    Some :class:`LabeledMutable` can be further simplified into multiple leaf mutables.
    In the current implementation, there are basically two kinds of :class:`LabeledMutable`:

    1. :class:`MutableSymbol`. This is usually referred to as a "parameter". They produce a key-value in the sample.
    2. :class:`~nni.mutable.annotation.MutableAnnotation`. They function as some kind of hint,
       and do not generate a key-value in the sample. Sometimes they can also be simplified and
       the :class:`MutableSymbol` they depend on would appear in the simplified result.
    """

    label: str

    def leaf_mutables(self, is_leaf: Callable[[Mutable], bool]) -> Iterable[LabeledMutable]:
        if is_leaf(self):
            # By default, is_leaf is true for MutableSymbol, and false for MutableAnnotation.
            # So MutableAnnotation must implement `is_leaf`, even if it decides to yield itself.
            yield self
        else:
            raise ValueError(f'is_leaf() should return True for this type of mutable: {type(self)}')

    def default(self, memo: Sample | None = None) -> Any:
        raise NotImplementedError(f'default() is not implemented for {self.__class__}')

    def random(self, memo: Sample | None = None, random_state: RandomState | None = None) -> Any:
        raise NotImplementedError(f'random() is not implemented in {self.__class__}.')

    def grid(self, memo: Sample | None = None, granularity: int | None = None) -> Iterable[Any]:
        raise NotImplementedError(f'grid() is not implemented in {self.__class__}.')

class Mutator(LabeledMutable):
    """
    Mutates graphs in model to generate new model.

    By default, mutator simplifies to a single-value dict with its own label as key, and itself as value.
    At freeze, the strategy should provide a :class:`MutationSampler` in the dict.
    This is because the freezing of mutator is dynamic
    (i.e., requires a variational number of random numbers, dynamic ranges for each random number),
    and the :class:`MutationSampler` here can be considered as some random number generator
    to produce a random sequence based on the asks in :meth:`Mutator.mutate`.

    On the other hand, a subclass mutator should implement :meth:`Mutator.mutate`, which calls :meth:`Mutator.choice` inside,
    and :meth:`Mutator.choice` invokes the bounded sampler to "random" a choice.

    The label of the mutator in most cases is the label of the nodes on which the mutator is applied to.

    I imagine that mutating any model space (other than graph) might be useful,
    but we would postpone the support to when we actually need it.
    """

    def __init__(self, *, sampler: Optional[MutationSampler] = None, label: Optional[str] = None):
        self.sampler: Optional[MutationSampler] = sampler
        self.label: str = auto_label(label)
        self.model: Optional[GraphModelSpace] = None
        self._cur_model: Optional[GraphModelSpace] = None
        self._cur_choice_idx: Optional[int] = None

    def extra_repr(self) -> str:
        return f'label={self.label!r}'

    def leaf_mutables(self, is_leaf: Callable[[Mutable], bool]) -> Iterable[LabeledMutable]:
        """By default, treat self as a whole labeled mutable in the format dict.

        Sub-class can override this to dry run the mutation upon the model and return the mutated model
        for the followed-up dry run.

        See Also
        --------
        nni.mutable.Mutable.leaf_mutables
        """
        # Same as `leaf_mutables` in LabeledMutable.
        return super().leaf_mutables(is_leaf)

    def check_contains(self, sample: Sample) -> SampleValidationError | None:
        """Check if the sample is valid for this mutator.

        See Also
        --------
        nni.mutable.Mutable.check_contains
        """
        if self.label not in sample:
            return SampleMissingError(f"Mutator {self.label} not found in sample.")
        if not isinstance(sample[self.label], MutationSampler):
            return SampleValidationError(f"Mutator {self.label} is not a MutationSampler.")
        return None

    def freeze(self, sample: dict[str, Any]) -> GraphModelSpace:
        """When freezing a mutator, we need a model to mutate on, as well as a sampler to generate choices.

        As how many times the mutator is applied on the model is often variational,
        a sample with fixed length will not work.
        The dict values in ``sample`` should be a sampler inheriting :class:`MutationSampler`.
        But there are also cases where ``simplify()`` converts the mutation process into some fixed operations
        (e.g., in :class:`StationaryMutator`).
        In this case, sub-class should handle the freeze logic on their own.

        :meth:`Mutator.freeze` needs to be called in a ``bind_model`` context.
        """
        self.validate(sample)
        assert self.model is not None, 'Mutator must be bound to a model before freezing.'
        return self.bind_sampler(sample[self.label]).apply(self.model)

    def bind_sampler(self, sampler: MutationSampler) -> Mutator:
        """Set the sampler which will handle :meth:`Mutator.choice` calls."""
        self.sampler = sampler
        return self

    @contextmanager
    def bind_model(self, model: GraphModelSpace) -> Iterator[Mutator]:
        """Mutators need a model, based on which they generate new models.
        This context manager binds a model to the mutator, and unbinds it after the context.

        Examples
        --------
        >>> with mutator.bind_model(model):
        ...     mutator.simplify()
        """
        try:
            self.model = model
            yield self
        finally:
            self.model = None

    def apply(self, model: GraphModelSpace) -> GraphModelSpace:
        """
        Apply this mutator on a model.
        The model will be copied before mutation and the original model will not be modified.

        Returns
        -------
        The mutated model.
        """
        assert self.sampler is not None
        copy = model.fork()
        copy.status = ModelStatus.Mutating
        self._cur_model = copy
        self._cur_choice_idx = 0
        self._cur_samples = []

        # Some mutate() requires a full mutation history of the model.
        # Therefore, parent needs to be set before the mutation.
        copy.parent = Mutation(self, self._cur_samples, model, copy)
        self.sampler.mutation_start(self, copy)
        self.mutate(copy)
        self.sampler.mutation_end(self, copy)
        self._cur_model = None
        self._cur_choice_idx = None
        return copy

    def mutate(self, model: GraphModelSpace) -> None:
        """
        Abstract method to be implemented by subclass.
        Mutate a model in place.
        """
        raise NotImplementedError()

    def choice(self, candidates: Iterable[Choice]) -> Choice:
        """Ask sampler to make a choice."""
        assert self.sampler is not None and self._cur_model is not None and self._cur_choice_idx is not None
        ret = self.sampler.choice(list(candidates), self, self._cur_model, self._cur_choice_idx)
        self._cur_samples.append(ret)
        self._cur_choice_idx += 1
        return ret

    def random(self, memo: Sample | None = None, random_state: RandomState | None = None) -> GraphModelSpace | None:
        """Use a :class:`_RandomSampler` that generates a random sample when mutates.

        See Also
        --------
        nni.mutable.Mutable.random
        """
        sample: Sample = {} if memo is None else memo
        if random_state is None:
            random_state = RandomState()
        if self.label not in sample:
            sample[self.label] = _RandomSampler(random_state)
        if self.model is not None:
            # Model is binded, perform the freeze.
            return self.freeze(sample)
        else:
            # This will only affect the memo.
            # Parent random will take care of the freeze afterwards.
            return None