"""Named-factor hybrid equities and domestic deterministic/GSR/SLV rates."""

import bisect
import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from dal_jax.errors import InvalidModelParameter, script_error
from dal_jax.index import CURRENCIES
from dal_jax.models.base import SampleDef, Scenario, validate_timeline
from dal_jax.models.correlated_bs import plain_equity
from dal_jax.models.gsr import GSR, covariance_factor, validate_correlation, varying_initial
from dal_jax.models.gsrslv import GSRSLV, integration_grid
from dal_jax.models.localvol import LocalVol, LocalVolSurface
from dal_jax.strings import ci_key


@dataclass(frozen=True, slots=True, kw_only=True)
class HybridBSEquity:
    name: str
    index: str
    currency: str
    factor: str
    spot: float
    vol: float
    div: float = 0.

    def __post_init__(self):
        object.__setattr__(self,"index",plain_equity(self.index))
        self.validate_params(self.default_params())

    @property
    def param_labels(self):
        return tuple(f"{label}:{self.index}" for label in ("spot","vol","div"))

    def default_params(self):
        return dict(zip(self.param_labels,map(jnp.asarray,(self.spot,self.vol,self.div))))

    def validate_params(self,params):
        spot,vol,div = (float(params[name]) for name in self.param_labels)
        if not all(map(math.isfinite,(spot,vol,div))) or spot <= 0. or vol < 0.:
            raise InvalidModelParameter("hybrid equity spot must be positive, volatility nonnegative and dividend finite")


@dataclass(frozen=True, slots=True, kw_only=True)
class HybridLocalVolEquity:
    name: str
    index: str
    currency: str
    factor: str
    spot: float
    surface: LocalVolSurface
    div: float = 0.
    max_step: float = 1./12.

    def __post_init__(self):
        object.__setattr__(self,"index",plain_equity(self.index))
        self._local()

    def _local(self):
        return LocalVol(name=self.name,index=self.index,currency=self.currency,factor=self.factor,spot=self.spot,
                        div=self.div,surface=self.surface,max_step=self.max_step)

    @property
    def param_labels(self):
        return self._local().param_labels[:-1]

    def default_params(self):
        return {name:value for name,value in self._local().default_params().items() if not name.startswith("rate:")}

    def validate_params(self,params):
        self._local().validate_params(dict(params,**{f"rate:{self.currency}":0.}))

    def vol_grid(self,params):
        return jnp.stack([jnp.stack([params[f"lvol:{self.index}:{i}:{j}"] for j in range(len(self.surface.times))])
                          for i in range(len(self.surface.spots))])


@dataclass(frozen=True, slots=True, kw_only=True)
class HybridDeterministicRate:
    name: str
    currency: str
    rate: float

    @property
    def param_labels(self):
        return (f"rate:{self.currency}",)

    def default_params(self):
        return {self.param_labels[0]:jnp.asarray(self.rate)}

    def log_df(self,params,time):
        return -params[self.param_labels[0]]*time


def interpolation_weights(times,time,scheme):
    n = len(times)
    weights = np.zeros(n)
    if time > times[-1]:
        excess = (time-times[-1])/(times[-1]-times[-2])
        weights[-2:] = -excess,1.+excess
        return weights
    if scheme == "LOG_LINEAR" or (scheme == "MIXED" and time <= times[max(1,n-5)]):
        hi = bisect.bisect_left(times,time)
        if hi == 0:
            weights[0] = 1.
        else:
            upper = (time-times[hi-1])/(times[hi]-times[hi-1])
            weights[hi-1:hi+1] = 1.-upper,upper
        return weights
    offset = max(1,n-5) if scheme == "MIXED" else 0
    return _natural_weights(times,time,offset)


def _natural_weights(times,time,offset):
    weights = np.zeros(len(times))
    axis = np.asarray(times[offset:]); count = len(axis)
    widths = np.diff(axis)
    system = np.diag(2.*(widths[:-1]+widths[1:]))
    if count > 3:
        system += np.diag(widths[1:-1],1)+np.diag(widths[1:-1],-1)
    rhs = np.zeros((count-2,count))
    for row in range(count-2):
        rhs[row,row:row+3] = 6./widths[row],-6./widths[row]-6./widths[row+1],6./widths[row+1]
    second = np.zeros((count,count)); second[1:-1] = np.linalg.solve(system,rhs)
    hi = min(max(bisect.bisect_left(axis,time),1),count-1)
    upper = (time-axis[hi-1])/widths[hi-1]; lower = 1.-upper
    tail = -lower*upper*widths[hi-1]**2/6.*((1.+lower)*second[hi-1]+(1.+upper)*second[hi])
    tail[hi-1] += lower; tail[hi] += upper
    weights[offset:] = tail
    return weights


@dataclass(frozen=True, slots=True, kw_only=True)
class HybridLogDfRate:
    name: str
    currency: str
    times: tuple
    log_df_values: tuple
    scheme: str = "LOG_LINEAR"

    def __post_init__(self):
        times,values = tuple(map(float,self.times)),tuple(map(float,self.log_df_values))
        minimum = {"LOG_LINEAR":2,"LOG_CUBIC_NATURAL":3,"MIXED":4}.get(self.scheme,math.inf)
        if (len(times) < minimum or len(times) != len(values) or times[0] != 0. or values[0] != 0. or
                not np.isfinite(times+values).all() or np.any(np.diff(times) <= 0.)):
            raise script_error("InvalidHybridCurve: finite increasing times and logDF values must start at zero and support the interpolation scheme")
        object.__setattr__(self,"times",times); object.__setattr__(self,"log_df_values",values)

    @property
    def param_labels(self):
        return tuple(f"logdf:{self.currency}:{i}" for i in range(1,len(self.times)))

    def default_params(self):
        return dict(zip(self.param_labels,map(jnp.asarray,self.log_df_values[1:])))

    def log_df(self,params,time):
        weights = interpolation_weights(self.times,time,self.scheme)
        return sum(weight*params[label] for weight,label in zip(weights[1:],self.param_labels))


@dataclass(frozen=True, slots=True, kw_only=True)
class HybridGSRRate:
    name: str
    model: GSR
    factors: tuple

    @property
    def currency(self):
        return self.model.curve.currency

    @property
    def param_labels(self):
        return self.model.param_labels

    def default_params(self):
        return self.model.default_params()


@dataclass(frozen=True, slots=True, kw_only=True)
class HybridGSRSLVRate:
    name: str
    model: GSRSLV
    vol_factor: str
    bridge_factor: str

    @property
    def currency(self):
        return self.model.gaussian.curve.currency

    @property
    def factors(self):
        return self.model.gaussian.vol.factor_names+(self.vol_factor,self.bridge_factor)

    @property
    def param_labels(self):
        return self.model.param_labels

    def default_params(self):
        return self.model.default_params()


@dataclass(frozen=True, slots=True, kw_only=True)
class HybridCorrelation:
    factor_names: tuple
    correlations: tuple
    name: str = "correlation"

    def __post_init__(self):
        names = tuple(self.factor_names)
        if not names or any(not name for name in names) or len(set(map(ci_key,names))) != len(names):
            raise script_error("InvalidHybridCorrelation: factor names must be nonempty and unique")
        matrix = validate_correlation(self.correlations,len(names),"InvalidHybridCorrelation")
        if np.linalg.eigvalsh(matrix).min() <= 1e-14:
            raise script_error("InvalidHybridCorrelation: matrix must be positive definite")
        object.__setattr__(self,"factor_names",names); object.__setattr__(self,"correlations",matrix)


def assemble_correlation(components,links=(),name="correlation"):
    names = tuple(sorted((factor for component in components for factor in component_factors(component)),key=ci_key))
    matrix = np.eye(len(names))
    blocks = {}
    for component in components:
        factors = component_factors(component)
        block = _component_correlation(component)
        for i,a in enumerate(factors):
            blocks[a] = component.name
            for j,b in enumerate(factors):
                matrix[names.index(a),names.index(b)] = block[i,j]
    _install_links(matrix,names,blocks,links)
    return HybridCorrelation(name=name,factor_names=names,correlations=matrix)


def _component_correlation(component):
    if isinstance(component,HybridGSRRate):
        return np.asarray(component.model.correlations)
    block = np.eye(len(component_factors(component)))
    if isinstance(component,HybridGSRSLVRate):
        block[:-1,:-1] = component.model.driver_correlations()
    return block


def _install_links(matrix,names,blocks,links):
    seen = set()
    for a,b,value in links:
        pair = frozenset((a,b))
        if a not in blocks or b not in blocks or blocks[a] == blocks[b] or pair in seen:
            raise script_error("InvalidHybridCorrelation: links must join distinct components once using registered factors")
        seen.add(pair)
        matrix[names.index(a),names.index(b)] = matrix[names.index(b),names.index(a)] = value


def component_factors(component):
    if isinstance(component,(HybridBSEquity,HybridLocalVolEquity)):
        return (component.factor,)
    return tuple(component.factors) if isinstance(component,(HybridGSRRate,HybridGSRSLVRate)) else ()


@dataclass(frozen=True,slots=True)
class HybridPlan:
    times: tuple
    definitions: tuple
    grid: tuple
    sample_indices: tuple
    rate_plan: object
    rate_outputs: tuple
    max_observations: int
    max_discounts: int


class HybridState(NamedTuple):
    log_spots: object
    vols: tuple
    dividends: object
    initial_carry: object
    rate: object
    rate_loading: object
    rate_noise: object
    rate_std: object
    rate_bridge: object
    rate_drift: object
    lower: object
    params: object


@dataclass(frozen=True,slots=True,kw_only=True)
class Hybrid:
    domestic_currency: str
    components: tuple
    correlation: HybridCorrelation
    name: str = "hybrid"

    supports_bb = True
    max_observed_indices = 2**31-1
    max_output_slots_per_sample = 2**31-1

    def __post_init__(self):
        components = tuple(sorted(self.components,key=lambda c:ci_key(c.name)))
        self._validate_component_names(components)
        object.__setattr__(self,"components",components)
        self._validate_equities()
        self._validate_factor_registry()
        self._validate_rate_correlations()
        self.validate_params(self.default_params())

    def _validate_component_names(self,components):
        if not components or self.domestic_currency not in CURRENCIES:
            raise script_error("InvalidHybridModel: components and supported domestic currency are required")
        if any(not c.name or c.currency != self.domestic_currency for c in components):
            raise script_error("InvalidHybridCurrency: named components must use the domestic currency")
        if len({ci_key(c.name) for c in components}) != len(components):
            raise script_error("DuplicateHybridComponent: component names must be unique")

    def _validate_equities(self):
        if not self.equities or len(self.components) != len(self.equities)+1:
            raise script_error("InvalidHybridNumeraire: equities and exactly one domestic rate component are required")
        if len({ci_key(c.index) for c in self.equities}) != len(self.equities):
            raise script_error("DuplicateHybridObservable: equity indices must be unique")

    def _validate_factor_registry(self):
        factors = tuple(f for c in self.components for f in component_factors(c))
        if any(not f for f in factors) or len(set(map(ci_key,factors))) != len(factors):
            raise script_error("DuplicateHybridFactor: factor names must be nonempty and unique")
        if set(map(ci_key,factors)) != set(map(ci_key,self.correlation.factor_names)):
            raise script_error("InvalidHybridCorrelation: provider factors must match the hybrid factor registry")

    def _validate_rate_correlations(self):
        if isinstance(self.rate,HybridGSRRate):
            if len(self.rate.factors) != self.rate.model.n_factors:
                raise script_error("InvalidHybridFactor: rate factor names must match the Gaussian kernel")
            expected = self.rate.model.correlations
            self._check_block(self.rate.factors,expected)
        elif isinstance(self.rate,HybridGSRSLVRate):
            factors = self.rate.factors
            expected = np.eye(len(factors)); expected[:-1,:-1] = self.rate.model.driver_correlations()
            self._check_block(factors,expected)
            bridge = self.factor_names.index(self.rate.bridge_factor)
            if np.any(np.abs(np.delete(self.ordered_correlation()[bridge],bridge)) > 1e-10):
                raise script_error("InvalidGSRSLVHybrid: bridge factor must be independent of all other factors")

    def _check_block(self,factors,expected):
        ids = [self.factor_names.index(factor) for factor in factors]
        if not np.allclose(self.ordered_correlation()[np.ix_(ids,ids)],expected,rtol=0.,atol=1e-10):
            raise script_error("InvalidHybridCorrelation: rate factor correlations must match the rate kernel")

    @property
    def equities(self):
        return tuple(c for c in self.components if isinstance(c,(HybridBSEquity,HybridLocalVolEquity)))

    @property
    def rate(self):
        return next(c for c in self.components if not isinstance(c,(HybridBSEquity,HybridLocalVolEquity)))

    @property
    def num_assets(self):
        return len(self.equities)

    @property
    def factor_names(self):
        return tuple(sorted((f for c in self.components for f in component_factors(c)),key=ci_key))

    @property
    def n_factors(self):
        return len(self.factor_names)+isinstance(self.rate,HybridGSRRate)

    @property
    def evaluation_date(self):
        return self.rate.model.evaluation_date if isinstance(self.rate,(HybridGSRRate,HybridGSRSLVRate)) else None

    @property
    def numeraire_is_deterministic(self):
        return self.rate.model.numeraire_is_deterministic if isinstance(self.rate,(HybridGSRRate,HybridGSRSLVRate)) else True

    @property
    def supports_discount_factors(self):
        return not isinstance(self.rate,HybridGSRSLVRate)

    @property
    def param_labels(self):
        return tuple(label for c in self.components for label in c.param_labels)

    def default_params(self):
        return {name:value for c in self.components for name,value in c.default_params().items()}

    def validate_params(self,params):
        for component in self.components:
            if isinstance(component,(HybridBSEquity,HybridLocalVolEquity)):
                component.validate_params(params)
            elif isinstance(component,(HybridGSRRate,HybridGSRSLVRate)):
                component.model.validate_params(params)
            elif any(not math.isfinite(float(params[name])) for name in component.param_labels):
                raise InvalidModelParameter("hybrid curve/rate parameters must be finite")

    def ordered_correlation(self):
        names = list(map(ci_key,self.correlation.factor_names))
        ids = [names.index(ci_key(name)) for name in self.factor_names]
        return np.asarray(self.correlation.correlations)[np.ix_(ids,ids)]

    def supports_index(self,index):
        if any(ci_key(index.name) == ci_key(c.index) for c in self.equities):
            return True
        return isinstance(self.rate,(HybridGSRRate,HybridGSRSLVRate)) and self.rate.model.supports_index(index)

    def allocate(self,timeline,sample_defs):
        times = tuple(map(float,timeline))
        validate_timeline(self,times,sample_defs)
        grid = self._integration_grid(times)
        samples = tuple(grid.index(t) for t in times)
        rate_outputs = []
        definitions = [SampleDef() for _ in grid]
        for i,definition in zip(samples,sample_defs):
            names,outputs = self._output_requests(definition)
            definitions[i] = SampleDef(index_names=names,discount_mats=definition.discount_mats)
            rate_outputs.append(outputs)
        gaussian = self._gaussian_model()
        rate_plan = gaussian.allocate(grid,definitions) if gaussian is not None else None
        return HybridPlan(times,tuple(sample_defs),grid,samples,rate_plan,tuple(rate_outputs),
                          max(len(d.index_names) for d in sample_defs),max(len(d.discount_mats) for d in sample_defs))

    def _integration_grid(self,times):
        anchors = times
        max_step = min((c.max_step for c in self.equities if isinstance(c,HybridLocalVolEquity)),default=math.inf)
        anchors,max_step = self._rate_grid(anchors,max_step)
        return integration_grid(anchors,max_step,dated=self.evaluation_date is not None)

    def _rate_grid(self,anchors,max_step):
        last = anchors[-1]
        if isinstance(self.rate,HybridGSRRate):
            anchors += tuple(t for t in self.rate.model._knots(0.,last) if 0. < t < last)
        elif isinstance(self.rate,HybridGSRSLVRate):
            anchors += tuple(t for t in self.rate.model.breakpoints() if 0. < t < last)
            max_step = min(max_step,self.rate.model.settings.max_step)
        return anchors,max_step

    def _output_requests(self,definition):
        from dal_jax.index import parse_index
        names,outputs = [],[]
        for name in definition.index_names:
            if not self.supports_index(parse_index(name)):
                raise script_error(f"UnsupportedModelObservation: {name}")
            equity = next((j for j,c in enumerate(self.equities) if ci_key(c.index) == ci_key(name)),None)
            outputs.append(("eq",equity) if equity is not None else ("rate",len(names)))
            if equity is None:
                names.append(name)
        return tuple(names),tuple(outputs)

    def _gaussian_model(self):
        if isinstance(self.rate,HybridGSRRate):
            return self.rate.model
        if isinstance(self.rate,HybridGSRSLVRate):
            return self.rate.model.gaussian
        return None

    def sim_dim(self,plan):
        return (len(plan.grid)-1)*self.n_factors

    def log_df(self,params,time):
        if isinstance(self.rate,(HybridGSRRate,HybridGSRSLVRate)):
            gaussian = self.rate.model if isinstance(self.rate,HybridGSRRate) else self.rate.model.gaussian
            return gaussian.curve.log_df(params,time)
        return self.rate.log_df(params,time)

    def init(self,params,plan):
        carry = jnp.stack([self.log_df(params,a)-self.log_df(params,b) for a,b in zip(plan.grid,plan.grid[1:])]) if len(plan.grid)>1 else jnp.empty(0)
        rate_state = None
        bs,qs,stds,bridges,drifts = [],[],[],[],[]
        if isinstance(self.rate,HybridGSRRate):
            rates = self.rate.model
            rate_state = rates.init(params,plan.rate_plan)
            for i,(start,end) in enumerate(zip(plan.grid,plan.grid[1:])):
                b = rate_state.loading[i+1]
                g,_ = rates.pieces(params,start)
                std = g*math.sqrt(end-start)
                q = .5*b*std
                var = jnp.einsum("i,ij,j->",q,jnp.asarray(rates.correlations),q)
                positive = var>0.
                bridge = jnp.where(positive,jnp.sqrt(jnp.where(positive,var/3.,1.)),0.)
                old_v,_,old_c = rates.integrals(params,0.,start)
                drift = carry[i]+jnp.dot(b,old_c)+.5*jnp.einsum("i,ij,j->",b,old_v,b)+(2./3.)*var
                bs.append(b); qs.append(q); stds.append(std); bridges.append(bridge); drifts.append(drift)
        elif isinstance(self.rate,HybridGSRSLVRate):
            slv = self.rate.model
            rate_state = slv.prepare_steps(params,plan.grid,slv.gaussian.init(params,plan.rate_plan,hjm=True))
        return self._hybrid_state(params,carry,rate_state,(bs,qs,stds,bridges,drifts))

    def _hybrid_state(self,params,carry,rate_state,gaussian_fields):
        bs,qs,stds,bridges,drifts = gaussian_fields
        vols = tuple(c.vol_grid(params) if isinstance(c,HybridLocalVolEquity) else params[f"vol:{c.index}"] for c in self.equities)
        n = self.rate.model.n_factors if isinstance(self.rate,HybridGSRRate) else 1
        def stack(values,shape):
            return jnp.stack(values) if values else jnp.zeros((len(carry),)+shape)
        return HybridState(jnp.stack([jnp.log(params[f"spot:{c.index}"]) for c in self.equities]),vols,
                           jnp.stack([params[f"div:{c.index}"] for c in self.equities]),carry,rate_state,
                           stack(bs,(n,)),stack(qs,(n,)),stack(stds,(n,)),stack(bridges,()),stack(drifts,()),
                           covariance_factor(jnp.asarray(self.ordered_correlation())),params)

    def generate(self,state,plan,normals):
        gsr = isinstance(self.rate,HybridGSRRate)
        slv = isinstance(self.rate,HybridGSRSLVRate)
        n = self.rate.model.n_factors if gsr else (self.rate.model.gaussian.n_factors if slv else 1)
        slots = tuple(self.factor_names.index(f) for f in component_factors(self.rate))
        equity_slots = jnp.asarray([self.factor_names.index(c.factor) for c in self.equities])
        deterministic = self._live_numeraire_is_deterministic(state.params)
        def step(carry,data):
            logs,rate = carry
            x,y,variance,latent,log_n = rate
            i,z = data
            named = state.lower@z[:len(self.factor_names)]
            old_log_n = log_n
            if gsr:
                drivers = named[jnp.asarray(slots)]
                log_n = log_n+state.rate_drift[i]+jnp.dot(state.rate_loading[i],x)+jnp.dot(state.rate_noise[i],drivers)+state.rate_bridge[i]*z[-1]
                x = x+state.rate_std[i]*drivers
            elif slv:
                coefficients = tuple(field[i] for field in state.rate[1:10])
                x,y,variance,latent,log_n = self.rate.model.advance(state.rate,rate,coefficients,
                                                                  named[jnp.asarray(slots[:-1])],named[slots[-1]])
            else:
                log_n = log_n+state.initial_carry[i]
            adjustment = jnp.where(deterministic,0.,log_n-old_log_n-state.initial_carry[i])
            grid = jnp.asarray(plan.grid,dtype=logs.dtype)
            time,dt = grid[i],jnp.diff(grid)[i]
            vols = jnp.stack([c.surface.volatility(time,jnp.exp(logs[j]),vols=state.vols[j]) if isinstance(c,HybridLocalVolEquity)
                              else state.vols[j] for j,c in enumerate(self.equities)])
            std = vols*jnp.sqrt(dt)
            logs = logs+state.initial_carry[i]+adjustment-state.dividends*dt-.5*std*std+std*named[equity_slots]
            rate = x,y,variance,latent,log_n
            return (logs,rate),(logs,x,y,log_n)
        initial = jax.tree.map(lambda value:varying_initial(value,normals),
                               (state.log_spots,(jnp.zeros(n,dtype=state.log_spots.dtype),jnp.zeros((n,n),dtype=state.log_spots.dtype),
                                                jnp.asarray(1.,dtype=state.log_spots.dtype),jnp.asarray(1.,dtype=state.log_spots.dtype),
                                                jnp.asarray(0.,dtype=state.log_spots.dtype))))
        _,(logs,xs,ys,log_n) = jax.lax.scan(jax.checkpoint(step,prevent_cse=False),initial,
                                          (jnp.arange(len(plan.grid)-1),normals.reshape((-1,self.n_factors))))
        logs = jnp.concatenate((state.log_spots[None,:],logs))
        xs = jnp.concatenate((jnp.zeros((1,n),dtype=xs.dtype),xs)); ys = jnp.concatenate((jnp.zeros((1,n,n),dtype=ys.dtype),ys))
        log_n = jnp.concatenate((jnp.zeros(1,dtype=log_n.dtype),log_n))
        ids = jnp.asarray(plan.sample_indices)
        spots = jnp.exp(logs[ids])
        if plan.times[0] == 0.:
            spots = spots.at[0].set(jnp.stack([state.params[f"spot:{c.index}"] for c in self.equities]))
        rate_scenario = self._rate_scenario(state,plan,xs,log_n,ys)
        return self._scenario(state,plan,spots,log_n,ids,rate_scenario)

    def _rate_scenario(self,state,plan,xs,log_n,ys):
        if isinstance(self.rate,HybridGSRRate):
            return self.rate.model.observe(state.rate,plan.rate_plan,xs,log_n)
        if isinstance(self.rate,HybridGSRSLVRate):
            return self.rate.model.gaussian.observe(state.rate.rates,plan.rate_plan,xs,log_n,ys)
        return None

    def _scenario(self,state,plan,spots,log_n,ids,rate_scenario):
        rows,discounts = [],[]
        for sample,outputs in enumerate(plan.rate_outputs):
            row = [spots[sample,index] if kind == "eq" else rate_scenario.observations[plan.sample_indices[sample],index] for kind,index in outputs]
            row += [jnp.asarray(0.,dtype=spots.dtype)]*(plan.max_observations-len(row))
            rows.append(jnp.stack(row) if row else jnp.empty(0,dtype=spots.dtype))
            discount = self._sample_discount(state,plan,sample,rate_scenario,spots.dtype)
            discounts.append(jnp.concatenate((discount,jnp.ones(plan.max_discounts-len(discount),dtype=spots.dtype))))
        bank = self._bank_numeraire(state,plan,log_n,ids)
        return Scenario(spots[:,0],jnp.where(jnp.asarray([d.numeraire for d in plan.definitions]),bank,1.),jnp.stack(rows),jnp.stack(discounts))

    def _sample_discount(self,state,plan,sample,rate_scenario,dtype):
        if rate_scenario is not None:
            return rate_scenario.discounts[plan.sample_indices[sample]]
        maturities = plan.definitions[sample].discount_mats
        return jnp.stack([jnp.exp(self.log_df(state.params,maturity)-self.log_df(state.params,plan.times[sample]))
                          for maturity in maturities]) if maturities else jnp.empty(0,dtype=dtype)

    def _bank_numeraire(self,state,plan,log_n,ids):
        deterministic = jnp.exp(jnp.stack([-self.log_df(state.params,time) for time in plan.times])).astype(log_n.dtype)
        return jnp.where(self._live_numeraire_is_deterministic(state.params),deterministic,jnp.exp(log_n[ids]))

    def _live_numeraire_is_deterministic(self,params):
        if self.numeraire_is_deterministic:
            return jnp.asarray(True)
        values = jnp.stack([params[name] for name in self._gaussian_model()._labels("g")])
        return jnp.all(values == 0.)
