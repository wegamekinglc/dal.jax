"""Exercise dates, regression-feature binding and cash-flow validation."""

from dataclasses import dataclass, replace

from dal_jax.errors import script_error
from dal_jax.index import parse_index
from dal_jax.script import ast as A
from dal_jax.strings import ci_eq, ci_key


@dataclass(frozen=True, slots=True)
class RegressionFeature:
    name: str
    variable_index: int | None = None
    output_by_event: tuple[int, ...] = ()


def exercise_node(event):
    return next((node for node in event if isinstance(node, A.Exercise)), None)


def validate_exercise_dates(product, date, model):
    for day, event in zip(product.event_dates, product.events):
        node = exercise_node(event)
        if node is None:
            continue
        if day <= date:
            raise script_error(
                f"UnsupportedExerciseDate: event={day}; expected an exercise date strictly after {date}; {node.source.describe()}"
            )
        if model is None:
            raise script_error(
                "UnsupportedExecutionMode: EXERCISE requires model-aware preparation"
            )


def _resolve_feature(name, variables, model):
    if ci_key(name).startswith("VAR[") and name.endswith("]"):
        found = next(
            (i for i, variable in enumerate(variables.var_names) if ci_eq(variable, name[4:-1])),
            None,
        )
        if found is None:
            raise script_error(f"UnknownLsmcRegressionVariable: {name}")
        return RegressionFeature(f"VAR[{variables.var_names[found]}]", variable_index=found)
    index = parse_index(name)
    if index.kind not in ("EQ", "IR") or not model.supports_index(index):
        raise script_error(
            f"UnsupportedLsmcRegressionFeature: {name}; expected a model-supported EQ or IR index"
        )
    return RegressionFeature(index.name)


def _feature_names(product, settings, model):
    names = settings.regression_features
    assets = getattr(model, "num_assets", 1)
    if not names and assets == 0:
        raise script_error(
            "MissingLsmcRegressionFeature: rate-model exercise requires product.regression_features"
        )
    if not names and assets > 1:
        if not settings.default_index:
            raise script_error(
                "AmbiguousLsmcRegressor: multi-asset exercise requires product.default_index"
            )
        names = (settings.default_index,)
    _validate_feature_names(names)
    return names


def _validate_feature_names(names):
    if len(names) > 3:
        raise script_error("InvalidLsmcFeatureBudget: supports at most three regression features")
    if any(not name for name in names):
        raise script_error("InvalidLsmcRegressionFeature: empty feature name")


def configure_regression(product, settings, model, plan):
    if not product.has_exercise:
        return plan, ()
    features = tuple(
        _resolve_feature(name, product.vars, model)
        for name in _feature_names(product, settings, model)
    )
    if len({ci_key(feature.name) for feature in features}) != len(features):
        raise script_error("DuplicateLsmcRegressionFeature: regression features must be distinct")
    if not features:
        return plan, (RegressionFeature("SPOT"),)
    definitions, features = bind_regression_outputs(
        product.events, plan.event_to_sample, plan.definitions, features
    )
    return replace(plan, definitions=definitions), features


def _feature_output(feature, event, sample, definitions):
    if feature.variable_index is not None or feature.name == "SPOT" or exercise_node(event) is None:
        return 0
    names = definitions[sample].index_names
    slot = next((i for i, name in enumerate(names) if ci_eq(name, feature.name)), len(names))
    if slot == len(names):
        definitions[sample] = replace(definitions[sample], index_names=names + (feature.name,))
    return slot


def bind_regression_outputs(events, event_to_sample, definitions, features):
    definitions = list(definitions)
    bound = []
    for feature in features:
        outputs = tuple(
            _feature_output(feature, event, sample, definitions)
            for event, sample in zip(events, event_to_sample)
        )
        bound.append(
            replace(feature, output_by_event=outputs) if feature.name != "SPOT" else feature
        )
    return tuple(definitions), tuple(bound)


def _validate_assignment(node, payoff_index, historical, paid):
    if isinstance(node, A.If):
        left = _validate_range(node.then_branch, payoff_index, historical, paid)
        right = _validate_range(node.else_branch, payoff_index, historical, paid)
        return left or right
    if isinstance(node, A.Pays):
        return paid or (not historical and node.args[0].index == payoff_index)
    return _validate_receiver_assignment(node, payoff_index, historical, paid)


def _validate_receiver_assignment(node, payoff_index, historical, paid):
    if isinstance(node, A.Assign) and node.args[0].index == payoff_index:
        zero = isinstance(node.args[1], A.Const) and node.args[1].const_val == 0.0
        if paid or not zero:
            raise script_error(
                "UnsupportedExercisePayoff: payoff receiver permits only literal-zero initialization before PAYS"
            )
        return paid
    return _validate_range(node.args, payoff_index, historical, paid)


def _validate_range(nodes, payoff_index, historical, paid):
    for node in nodes:
        paid = _validate_assignment(node, payoff_index, historical, paid)
    return paid


def validate_payoff(product, initial_values):
    if not product.has_pays:
        return
    index = product.payoff_index
    if initial_values[index] != 0.0:
        raise script_error(
            "UnsupportedExercisePayoff: EXERCISE requires a zero initial payoff receiver"
        )
    for events, historical in ((product.past_events, True), (product.events, False)):
        _validate_range(tuple(node for event in events for node in event), index, historical, False)


class Liveness:
    """Backward roots are exercises, selected payments and regression variables."""

    def __init__(self, payoff_index, features):
        self.payoff = payoff_index
        self.regression_variables = {
            feature.variable_index for feature in features if feature.variable_index is not None
        }
        self.live = {payoff_index} if payoff_index >= 0 else set()
        self.handlers = {
            A.Assign: self.assignment,
            A.Pays: self.payment,
            A.Exercise: self.exercise,
            A.If: self.branch,
            A.Collect: self.collect,
        }

    def expressions(self, node):
        self.live.update(child.index for child in A.walk(node) if isinstance(child, A.Var))

    def event(self, event):
        kept = []
        for node in reversed(event):
            result = self.statement(node)
            if result is not None:
                kept.append(result)
        return tuple(reversed(kept))

    def statement(self, node):
        return self.handlers.get(type(node), self.generic)(node)

    def assignment(self, node):
        index = node.args[0].index
        if index not in self.live:
            return None
        self.live.discard(index)
        self.expressions(node.args[1])
        return node

    def payment(self, node):
        if node.args[0].index != self.payoff and node.args[0].index not in self.live:
            return None
        self.expressions(node.args[1])
        return node

    def exercise(self, node):
        self.live.update(self.regression_variables)
        return self.generic(node)

    def collect(self, node):
        children = self.event(node.args)
        return node.with_args(children) if children else None

    def generic(self, node):
        self.expressions(node)
        return node

    def branch(self, node):
        after = self.live.copy()
        right = self.event(node.else_branch)
        right_live = self.live
        self.live = after
        left = self.event(node.then_branch)
        self.live |= right_live
        if not left and not right:
            return None
        self.expressions(node.condition)
        return replace(
            node,
            args=(node.condition,) + left + right,
            first_else=1 + len(left) if node.has_else else -1,
        )


def prune_lsmc(events, payoff_index, features):
    processor = Liveness(payoff_index, features)
    return tuple(reversed(tuple(processor.event(event) for event in reversed(events))))
