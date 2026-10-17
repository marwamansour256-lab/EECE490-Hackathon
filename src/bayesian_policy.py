"""Adaptive inventory decisions with a shared Gamma-Poisson demand state.

Early demand E|Z ~ Poisson(mu_early Z), late demand L|Z ~ Poisson(mu_late Z),
Z ~ Gamma(shape, rate=shape). If opening stock q sells out, observe E >= q,
not E == q. The dynamic program integrates that censored posterior explicitly.
shape=None is the independent Poisson control. All distributions are discrete
with tails below a declared numerical tolerance; no realized late demand enters.
"""
import numpy as np
from scipy.stats import nbinom, poisson


def distribution(mean, shape=None):
    if mean < 0 or not np.isfinite(mean) or (shape is not None and (shape <= 0 or not np.isfinite(shape))):
        raise ValueError('Invalid demand distribution')
    return poisson(mean) if shape is None else nbinom(shape, shape/(shape+mean))


def posterior_parameters(early_mean, late_mean, observed_early, shape):
    if shape is None:
        return float(late_mean), None
    updated_shape = shape + observed_early
    updated_mean = late_mean*updated_shape/(shape+early_mean)
    return float(updated_mean), float(updated_shape)


def joint_demand(early_mean, late_mean, shape=None, adaptive=True, tolerance=1e-9):
    early_dist, late_dist = distribution(early_mean, shape), distribution(late_mean, shape)
    emax = int(early_dist.ppf(1-tolerance))+2
    lmax = int(late_dist.ppf(1-tolerance))+2
    if max(emax, lmax) > 1000:
        raise ValueError('Scenario exceeds numerical support limit; aggregate or revise numerical method')
    early = np.arange(emax+1)
    late = np.arange(lmax+1)
    ep = early_dist.pmf(early)
    if shape is None or not adaptive:
        joint = ep[:, None]*late_dist.pmf(late)[None, :]
    else:
        updated_shape = shape+early
        probability = (shape+early_mean)/(shape+early_mean+late_mean)
        conditional = nbinom.pmf(late[None, :], updated_shape[:, None], probability)
        joint = ep[:, None]*conditional
    retained_mass = float(joint.sum())
    if retained_mass < 1-5*tolerance:
        raise ArithmeticError('Demand support truncation exceeds tolerance')
    joint /= retained_mass
    return joint, 1-retained_mass


def conditional_late(joint, observed_early, sold_out=False):
    """Return P(L|E=e) or P(L|E>=e); sold-out zero stock conveys no demand info."""
    if observed_early < 0 or observed_early != int(observed_early) or observed_early >= joint.shape[0]:
        raise ValueError('Early observation outside support')
    masses = joint[observed_early:].sum(axis=0) if sold_out else joint[observed_early].copy()
    mass = float(masses.sum())
    if mass <= 0:
        raise ValueError('Observation has zero probability under the demand model')
    return masses/mass


def sales_curve(pmf, max_stock):
    demand = np.arange(len(pmf))
    cumulative = np.cumsum(pmf)
    cumulative_value = np.cumsum(demand*pmf)
    stock = np.arange(max_stock+1)
    at = np.minimum(stock, len(pmf)-1)
    return cumulative_value[at]+stock*np.maximum(0., 1-cumulative[at])


def choose_topup(pmf, remaining, price, cost, setup, premium, salvage, fill_floor):
    # More than maximum supported demand cannot improve profit if salvage < cost.
    cap = max(0, len(pmf)-1-remaining)
    orders = np.arange(cap+1)
    stock = remaining+orders
    expected_sales = sales_curve(pmf, int(stock[-1]))[stock]
    mean = float(np.dot(np.arange(len(pmf)), pmf))
    values = price*expected_sales+salvage*(stock-expected_sales)-(cost+premium)*orders-setup*(orders>0)
    values[expected_sales+1e-10 < fill_floor*mean] = -np.inf
    best = int(np.argmax(values))
    return int(orders[best]), float(values[best])


def optimize_adaptive(mean, early_share, price, cost, shape=None, adaptive=True,
                      setup=.25, premium=.10, salvage_fraction=0., fill_floor=.90,
                      use_observation=True):
    numbers = [mean, early_share, price, cost, setup, premium, salvage_fraction, fill_floor]
    if not (np.isfinite(numbers).all() and mean >= 0 and 0 <= early_share <= 1 and price > cost > 0
            and setup >= 0 and premium >= 0 and 0 <= salvage_fraction < 1 and 0 <= fill_floor < 1):
        raise ValueError('Invalid economic or forecast input')
    early_mean, late_mean = mean*early_share, mean*(1-early_share)
    joint, omitted = joint_demand(early_mean, late_mean, shape, adaptive)
    marginal_late = joint.sum(axis=0)
    ep = joint.sum(axis=1)
    egrid = np.arange(len(ep))
    salvage = salvage_fraction*cost
    # Opening quantity may cover late demand too, so use total supported demand.
    cap = joint.shape[0]+joint.shape[1]-2
    early_sales = sales_curve(ep, cap)
    valid_first = np.where(early_sales+1e-10 >= fill_floor*early_mean)[0]
    # Precompute late value for each exact early observation and remaining stock.
    values_by_e = np.zeros((len(ep), cap+1))
    orders_by_e = np.zeros((len(ep), cap+1), dtype=int)
    for e in range(len(ep)):
        if ep[e] <= 0:
            continue
        pmf = joint[e]/ep[e]
        remaining = np.arange(cap+1)
        orders = np.arange(len(pmf))
        stocks = remaining[:, None]+orders[None, :]
        sales = sales_curve(pmf, cap+len(pmf)-1)[stocks]
        actual_values = price*sales+salvage*(stocks-sales)-(cost+premium)*orders[None, :]-setup*(orders[None, :]>0)
        decision_pmf = pmf if use_observation else marginal_late
        decision_sales = sales_curve(decision_pmf, cap+len(pmf)-1)[stocks]
        decision_values = price*decision_sales+salvage*(stocks-decision_sales)-(cost+premium)*orders[None, :]-setup*(orders[None, :]>0)
        decision_values[decision_sales+1e-10 < fill_floor*np.dot(np.arange(len(pmf)), decision_pmf)] = -np.inf
        chosen = np.argmax(decision_values, axis=1)
        values_by_e[e] = actual_values[remaining, chosen]
        orders_by_e[e] = chosen
    best_value = -np.inf
    best = None
    for opening in valid_first:
        exact = egrid[egrid < opening]
        value = price*early_sales[opening]-cost*opening
        value += float(np.dot(ep[exact], values_by_e[exact, opening-exact]))
        actions = {str(int(opening-e)): int(orders_by_e[e, opening-e]) for e in exact}
        tail_mass = float(ep[opening:].sum())
        if tail_mass > 0:
            pmf = joint[opening:].sum(axis=0)/tail_mass
            q, _ = choose_topup(pmf if use_observation else marginal_late, 0, price, cost, setup, premium, salvage, fill_floor)
            sales = float(sales_curve(pmf, q)[q])
            v = price*sales+salvage*(q-sales)-(cost+premium)*q-setup*(q>0)
            value += tail_mass*v
            actions['0'] = q
        else:
            actions['0'] = 0
        if value > best_value:
            best_value = value
            best = dict(opening_units=int(opening), replenish_by_remaining_stock=actions)
    # Same unconditional daily distribution and daily expected-fill comparator.
    total_pmf = np.zeros(cap+1)
    for e in range(joint.shape[0]):
        total_pmf[e:e+joint.shape[1]] += joint[e]
    one_q, one_value = choose_topup(total_pmf, 0, price, cost, 0., 0., salvage, fill_floor)
    best.update(mean=mean, mean_early=early_mean, mean_late=late_mean, shape=shape,
                adaptive=adaptive, use_observation=use_observation, price=price, cost=cost, setup=setup, premium=premium,
                salvage=salvage, fill_floor=fill_floor, one_batch_units=one_q,
                expected_two_stage_contribution=best_value, expected_one_batch_contribution=one_value,
                omitted_probability_mass=omitted)
    return best


def realize_adaptive(policy, early_demand, late_demand):
    if any(not np.isfinite(x) or x < 0 or int(x) != x for x in [early_demand, late_demand]):
        raise ValueError('Demand must be nonnegative integer counts')
    sold_early = min(early_demand, policy['opening_units'])
    remaining = int(policy['opening_units']-sold_early)
    topup = policy['replenish_by_remaining_stock'][str(remaining)]
    sold_late = min(late_demand, remaining+topup)
    unsold = remaining+topup-sold_late
    contribution = policy['price']*(sold_early+sold_late)+policy['salvage']*unsold-policy['cost']*policy['opening_units']-(policy['cost']+policy['premium'])*topup-policy['setup']*(topup>0)
    demand = early_demand+late_demand
    oneq = policy['one_batch_units']
    oneprofit = policy['price']*min(demand, oneq)+policy['salvage']*max(oneq-demand, 0)-policy['cost']*oneq
    return dict(opening_units=policy['opening_units'], replenished_units=topup,
                sold_units=sold_early+sold_late, unsold_units=unsold,
                unmet_units=demand-sold_early-sold_late, contribution=contribution,
                one_batch_contribution=oneprofit, incremental_contribution=contribution-oneprofit,
                early_sold_out=bool(early_demand >= policy['opening_units']))
