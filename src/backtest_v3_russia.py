from datetime import datetime, timedelta
from IPython.display import Image
from scipy.ndimage import shift
import matplotlib.pyplot as plt
from scipy.stats import cauchy
import scipy.stats as stats
from datetime import date

# from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib
import random
import time
import math
import warnings

warnings.filterwarnings("ignore")


def range_prices(pa_t, pb_t, N_t):
    """
    Create price ranges for buckets.

    Args:
    pa_t (float): Lower price limit
    pb_t (float): Upper price limit
    N_t (int): Number of buckets

    Returns:
    tuple: (bs_Bi, as_Bi, pr_ranges)
        bs_Bi (np.array): Upper bounds of each bucket
        as_Bi (np.array): Lower bounds of each bucket
        pr_ranges (np.array): All price ranges including upper and lower limits
    """
    # Calculate bucket sizes
    bs_Bi = np.arange(pb_t, pa_t, -(pb_t - pa_t) / N_t)

    # Shift array to get lower bounds
    as_Bi = np.roll(bs_Bi, -1)
    as_Bi[-1] = pa_t

    # Create full range of prices
    pr_ranges = np.append(pb_t, as_Bi)

    return bs_Bi, as_Bi, pr_ranges


def z_list_t(tau, prices_c, ind_in_):
    """
    Determine time instants when the reference bucket Z(s) needs to be redefined.

    Args:
    tau (int): Constant threshold for bucket difference
    prices_c (np.array): Set of prices
    ind_in_ (np.array): Bucket indices for each price

    Returns:
    np.array: Indices where the reference bucket should be reset
    """
    z_list = []
    for i in range(len(prices_c)):
        if i == 0:
            z_list.append(i)
            z_ = i
        else:
            # Check if the current bucket is more than tau away from the reference bucket
            if abs(ind_in_[i] - ind_in_[z_]) > tau:
                z_list.append(i)
                z_ = i

    return np.asarray(z_list)


def x_Nd(tau, mu_bucket, p0, range_prices_, a):
    """
    Determine the set of shares (x_i) for capital allocation based on the reference bucket.

    Args:
    tau (int): Constant threshold
    mu_bucket (int): Number of buckets
    p0 (float): Current price
    range_prices_ (tuple): Price ranges
    a (float): Allocation strategy constant

    Returns:
    np.array: Allocation shares for each bucket
    """
    # Find the bucket index for the current price
    x_ind_Z = np.digitize(p0, range_prices_[2], right=True) - 1

    # Create allocation array
    xi_ar = np.where(
        (
            (np.tile(np.arange(mu_bucket), (len(p0), 1)) > x_ind_Z.reshape(-1, 1))
            & (np.tile(np.arange(mu_bucket), (len(p0), 1)) < x_ind_Z.reshape(-1, 1) + tau + 1)
        ),
        a / (2 * tau),
        np.where(
            (np.tile(np.arange(mu_bucket), (len(p0), 1)) < x_ind_Z.reshape(-1, 1))
            & (np.tile(np.arange(mu_bucket), (len(p0), 1)) > x_ind_Z.reshape(-1, 1) - tau - 1),
            a / (2 * tau),
            0,
        ),
    )

    # Adjust allocation for the reference bucket
    xi_ar[np.arange(len(p0)), x_ind_Z] = 1 - np.sum(xi_ar, axis=1)

    return xi_ar


def V_2d(price, range_prices_, V_m, L_cur):
    """
    Calculate the 2D state of liquidity positions for a given price.

    Args:
    price (float): Current price
    range_prices_ (array): Array of price ranges
    V_m (array): Initial state matrix
    L_cur (float): Current liquidity

    Returns:
    array: Updated state matrix
    """
    # Find the index of the current price bucket
    ind_in_ = np.digitize(price, range_prices_, right=True)

    # Update state for buckets below the current price
    V_m[1:ind_in_, 1] = 0  # Set y to 0 for lower buckets
    V_m[1:ind_in_, 0] = 1 / np.sqrt(V_m[1:ind_in_, 4]) - 1 / np.sqrt(
        V_m[1:ind_in_, 3]
    )  # Calculate x for lower buckets

    # Update state for buckets above the current price
    V_m[ind_in_ + 1 : -1, 1] = np.sqrt(V_m[ind_in_ + 1 : -1, 3]) - np.sqrt(
        V_m[ind_in_ + 1 : -1, 4]
    )  # Calculate y for upper buckets
    V_m[ind_in_ + 1 : -1, 0] = 0  # Set x to 0 for upper buckets

    # Update state for the current price bucket
    V_m[ind_in_, 1] = np.sqrt(price) - np.sqrt(V_m[ind_in_, 4])  # Calculate y for current bucket
    V_m[ind_in_, 0] = 1 / np.sqrt(price) - 1 / np.sqrt(
        V_m[ind_in_, 3]
    )  # Calculate x for current bucket

    # Set liquidity for all valid buckets
    V_m[1:-1, 2] = L_cur

    # Adjust x and y values based on liquidity
    V_m[:, :2] *= V_m[:, 2].reshape(-1, 1)

    return V_m


def V_3d(prices, range_prices_, V_m, L_cur):
    """
    Calculate the 3D state of liquidity positions for a range of prices.

    Args:
    prices (array): Array of prices
    range_prices_ (array): Array of price ranges
    V_m (array): Initial state matrix
    L_cur (float): Current liquidity

    Returns:
    array: 3D state matrix
    """
    # Find the indices of price buckets for all prices
    ind_in = np.digitize(prices, range_prices_, right=True)

    # Create a 3D matrix by repeating V_m for each price
    V_Bi2 = np.repeat(V_m[np.newaxis, :, :], len(ind_in), axis=0)

    # Determine which buckets are above, below, or at the current price for each sheet
    ind_up_ = V_Bi2[:, :, 5] < ind_in.reshape(-1, 1)
    ind_dwn_ = V_Bi2[:, :, 5] > ind_in.reshape(-1, 1)
    ind_in_ = V_Bi2[:, :, 5] == ind_in.reshape(-1, 1)

    # Update state for buckets above the current price
    V_Bi2[:, :, 1][ind_up_] = 0
    V_Bi2[:, :, 0][ind_up_] = 1 / np.sqrt(V_Bi2[:, :, 4][ind_up_]) - 1 / np.sqrt(
        V_Bi2[:, :, 3][ind_up_]
    )

    # Update state for buckets below the current price
    V_Bi2[:, :, 1][ind_dwn_] = np.sqrt(V_Bi2[:, :, 3][ind_dwn_]) - np.sqrt(V_Bi2[:, :, 4][ind_dwn_])
    V_Bi2[:, :, 0][ind_dwn_] = 0

    # Update state for the current price bucket
    V_Bi2[:, :, 1][ind_in_] = np.sqrt(prices) - np.sqrt(V_Bi2[:, :, 4][ind_in_])
    V_Bi2[:, :, 0][ind_in_] = 1 / np.sqrt(prices) - 1 / np.sqrt(V_Bi2[:, :, 3][ind_in_])

    # Set liquidity for all valid buckets
    V_Bi2[:, 1:-1, 2] = L_cur

    # Adjust x and y values based on liquidity
    V_Bi2[:, :, :2] *= V_Bi2[:, :, 2].reshape(-1, len(V_Bi2[:, :, 2][0]), 1)

    return V_Bi2


def ret_AB(V_3d_m, g, cur_pr, lim_dict, W_start, i_, LIM_LIQ=False, MCoef_adv=False):
    """
    Calculate returns and apply liquidity limits if specified.

    Args:
    V_3d_m (array): 3D state matrix
    g (float): Fee parameter
    cur_pr (float): Current price
    lim_dict (dict): Dictionary of liquidity limits
    W_start (float): Starting capital
    i_ (int): Current iteration
    LIM_LIQ (bool): Whether to apply liquidity limits
    MCoef_adv (bool): Whether to use advanced market coefficient

    Returns:
    tuple: (ret_ep, sum_ret_ep)
        ret_ep (array): Returns for each bucket
        sum_ret_ep (float): Total returns
    """
    # Calculate returns as the positive differences in state
    ret_ep = np.diff(V_3d_m[:, :, :2], axis=0)
    ret_ep = np.where(ret_ep > 0, ret_ep, 0)
    ret_ep = np.sum(ret_ep, axis=0)

    # Apply liquidity limits if specified
    if LIM_LIQ:
        ret_ep_ = np.sum(ret_ep, axis=0)
        vol_coef = (ret_ep_[0] * cur_pr + ret_ep_[1]) / (W_start * len(V_3d_m))
        if vol_coef > lim_dict[g]:
            ret_ep = ret_ep * (lim_dict[g] / vol_coef)

    # Apply advanced market coefficient if specified
    if MCoef_adv:
        ret_ep_ = np.sum(ret_ep, axis=0)
        vol_coef = (ret_ep_[0] * cur_pr + ret_ep_[1]) / (W_start * len(V_3d_m))
        if vol_coef > lim_dict[i_]:
            ret_ep = ret_ep * (lim_dict[i_] / vol_coef)

    return ret_ep, np.sum(ret_ep, axis=0) * g


# now let’s write the liquidity distribution function for capital allocation, -
# - so that settlements are not based on 1-unit liquidity, -
# - for given shares x and the current market price, which is fixed until the next reset
# The architecture of the function is such that it must be applied for each epoch separately, since -
# - W will change depending on the accumulated commissions - this will in any case entail -
# - sequential calculation + as an option it will be possible to fix commissions without allocation
# If the price hits the border, then we calculate the liquidity for the block above / below -
# - similar to liquidity only for x and y!


def Li_Ei(price_upd, W, xi_array, ind_in, z_moment, V_1):
    """
    Calculate liquidity distribution for capital allocation based on current market price.

    This function calculates the liquidity distribution for each price bucket, considering
    the given shares and the current market price. It's designed to be applied for each
    epoch separately, as W (total capital) may change due to accumulated fees.

    Args:
    price_upd (float): Current market price
    W (float): Total capital available for allocation
    xi_array (np.array): Array of allocation shares for each bucket
    ind_in (np.array): Array of bucket indices for each price
    z_moment (int): Index of the current moment in the price array
    V_1 (np.array): Array containing price range information for each bucket

    Returns:
    np.array: Array of liquidity values for each bucket
    """

    # Initialize array to store liquidity for each bucket
    L_ar = np.zeros(len(xi_array))

    # Calculate capital allocation for each bucket
    Wx = W * xi_array

    # Current price
    p_up = price_upd

    # Index of the bucket just below the current price
    ind_up = ind_in[z_moment] - 1

    # Calculate liquidity for buckets below the current price
    L_ar[:ind_up] = (
        Wx[:ind_up]
        / p_up
        * np.sqrt(V_1[:, 4][1 : ind_up + 1] * V_1[:, 3][1 : ind_up + 1])
        / (np.sqrt(V_1[:, 3][1 : ind_up + 1]) - np.sqrt(V_1[:, 4][1 : ind_up + 1]))
    )

    # Calculate liquidity for buckets above the current price
    L_ar[ind_up:] = Wx[ind_up:] / (
        np.sqrt(V_1[:, 3][ind_up + 1 : -1]) - np.sqrt(V_1[:, 4][ind_up + 1 : -1])
    )

    # Handle the case when the current price is not exactly at a bucket boundary
    if p_up != V_1[:, 3][ind_up + 1]:
        # Calculate x_ and y_ for the current price bucket
        x_ = (np.sqrt(V_1[:, 3][ind_up + 1]) * np.sqrt(p_up)) / (
            np.sqrt(V_1[:, 3][ind_up + 1]) - np.sqrt(p_up)
        )
        y_ = 1 / (np.sqrt(p_up) - np.sqrt(V_1[:, 4][ind_up + 1]))

        # Calculate liquidity for the current price bucket
        L_ar[ind_up] = Wx[ind_up] / (p_up + x_ / y_) * x_

    return L_ar


def ret_AB_(V_3d_m, g, cur_pr, lim_dict, W_start, i_, LIM_LIQ=False, MCoef_adv=False):

    ret_ep = np.diff(
        V_3d_m[:, :, :2], axis=0
    )  # taking into account distributed liquidity, we calculate changes in contract states from layer to layer
    ret_ep = np.where(
        ret_ep > 0, ret_ep, 0
    )  # We leave only positive values, as rewards are taken from the type of token contributed by the trader

    if LIM_LIQ:
        ret_ep_ = np.sum(
            np.sum(ret_ep, axis=0), axis=0
        )  # value of the fixed base for calculating remuneration
        vol_coef = (ret_ep_[0] * cur_pr + ret_ep_[1]) / (W_start * len(V_3d_m))

        if vol_coef > lim_dict[g]:
            ret_ep = ret_ep * (lim_dict[g] / vol_coef)

    if MCoef_adv:
        ret_ep_ = np.sum(
            np.sum(ret_ep, axis=0), axis=0
        )  # value of the fixed base for calculating remuneration
        vol_coef = (ret_ep_[0] * cur_pr + ret_ep_[1]) / (W_start * len(V_3d_m))

        if vol_coef > lim_dict[i_]:
            ret_ep = ret_ep * (lim_dict[i_] / vol_coef)

    return ret_ep


def V_init(mu_bucket, pa_Bi, pb_Bi):
    # 2D matrix with the number of buckets+2 to represent prices above/below the range
    V_Bi = np.zeros((mu_bucket + 2, 6))
    # Let's number the buckets. 0th and last hypothetical
    V_Bi[:, 5] = np.arange(0, mu_bucket + 2)
    V_Bi[1:-1, 4] = np.asarray(pa_Bi)  # upper limit of the bucket
    V_Bi[1:-1, 3] = np.asarray(pb_Bi)  # lower limit of the bucket
    V_Bi[1:-1, 2] = 0  # fixed liquidity

    return V_Bi


# main function
def v3_model(
    W_init,
    ret_reloc,
    gas_price,
    t_epoch,
    z_moments_time,
    price_set,
    xi_array,
    range_prices_func,
    ind_in_func,
    tau,
    low_memory=False,
    NO_cost=False,
    LIQ_fix=False,
    MCoef_adv=False,
    ConstW=False,
):
    """
    Simulate a Uniswap V3-like automated market maker (AMM) model.

    Args:
    W_init (float): Initial capital
    ret_reloc (bool): Whether to relocate returns
    gas_price (float): Gas price for transaction costs
    t_epoch (int): Number of epochs
    z_moments_time (array): Time moments for redefining reference buckets
    price_set (array): Set of prices
    xi_array (array): Liquidity distribution array
    range_prices_func (function): Function to calculate price ranges
    ind_in_func (function): Function to calculate bucket indices
    tau (int): Threshold for bucket difference
    low_memory (bool): Whether to use low memory mode
    NO_cost (bool): Whether to ignore transaction costs
    LIQ_fix (bool): Whether to use fixed liquidity
    MCoef_adv (bool): Whether to use advanced market coefficient
    ConstW (bool): Whether to keep capital constant

    Returns:
    tuple: Various metrics including final capital, returns, costs, etc.
    """

    final_returns_ = 0
    final_costs = 0
    W_ep_start = np.zeros([t_epoch + 1])
    cost_ep = np.zeros([t_epoch + 1])
    fee_ep = np.zeros([t_epoch])
    bh_str = np.zeros([t_epoch])
    ret_ep_st = []

    for i in range(t_epoch - 1):

        if i == 0:
            W_ = W_init
            if NO_cost:
                cost = 0
            else:
                cost = (
                    2
                    * 215000
                    * gas_price
                    * 10 ** (-9)
                    * price_set[z_moments_time[i]]
                    * (2 * tau + 1)
                )
            W_ -= cost

        else:
            if NO_cost:
                cost = 0
            else:
                dW_ep = W_ * xi_array[i] - W_ep_start[i - 1] * xi_array[i - 1]
                cost_Base_pos = np.sum(np.where(dW_ep[dW_ep > 0], 1, 0))
                cost_Base_neg = np.sum(np.where(dW_ep[dW_ep < 0], 1, 0))
                cost = (
                    215000
                    * gas_price
                    * 10 ** (-9)
                    * price_set[z_moments_time[i]]
                    * (2 * cost_Base_pos + cost_Base_neg)
                )
            W_ -= cost

        if W_ < 0:
            W_ = 0

        if ConstW:
            W_ = W_init

        final_costs += cost
        cost_ep[i] = cost
        W_ep_start[i] = W_

        V_2d_ep = V_2d(
            price_set[z_moments_time[i]],
            range_prices_func[2],
            V_init(mu_bucket, range_prices_func[1], range_prices_func[0]),
            1,
        )
        Li_ep = Li_Ei(
            price_set[z_moments_time[i]], W_, xi_array[i], ind_in_func, z_moments_time[i], V_2d_ep
        )
        V_3d_ep = V_3d(
            price_set[z_moments_time[i] : z_moments_time[i + 1] + 1],
            range_prices_func[2],
            V_init(mu_bucket, range_prices_func[1], range_prices_func[0]),
            Li_ep,
        )
        ret_ep = ret_AB(
            V_3d_ep,
            gamma,
            price_set[z_moments_time[i + 1]],
            fee_liq_dict,
            W_,
            i,
            LIQ_fix,
            MCoef_adv,
        )
        ret_ep_dW = ret_ep[1][0] * price_set[z_moments_time[i + 1]] + ret_ep[1][1]
        final_returns_ += ret_ep_dW
        fee_ep[i] = ret_ep_dW

        W_end_ep = np.sum(V_3d_ep[-1, 1:-1, :2], axis=0)
        W_end_ep = W_end_ep[0] * price_set[z_moments_time[i + 1]] + W_end_ep[1]
        W_ = W_end_ep

        W_end_ep_bh = np.sum(V_3d_ep[0, 1:-1, :2], axis=0)
        bh_str[i] = (W_end_ep_bh[0] * price_set[z_moments_time[i + 1]] + W_end_ep_bh[1]) / (
            W_end_ep_bh[0] * price_set[z_moments_time[i]] + W_end_ep_bh[1]
        ) - 1

        if i == 0:

            if low_memory:
                V3_total = np.stack([V_3d_ep[0], V_3d_ep[-1]], axis=0)
            else:
                V3_total = V_3d_ep

            Li_total = Li_ep
            ret_tab_total = ret_ep[0]
        else:

            if low_memory:
                V3_total = np.concatenate(
                    (V3_total, np.stack([V_3d_ep[0], V_3d_ep[-1]], axis=0)), axis=0
                )
            else:
                V3_total = np.concatenate((V3_total, V_3d_ep), axis=0)

            Li_total = np.concatenate((Li_total, Li_ep), axis=0)
            ret_tab_total = np.concatenate((ret_tab_total, ret_ep[0]), axis=0)

        if low_memory:
            pass
        else:
            ret_ep_st.append(
                ret_AB_(
                    V_3d_ep,
                    gamma,
                    price_set[z_moments_time[i + 1]],
                    fee_liq_dict,
                    W_,
                    i,
                    LIQ_fix,
                    MCoef_adv,
                )
            )

        if ret_reloc:
            W_ += ret_ep_dW

    if t_epoch > 1:

        if NO_cost:
            cost = 0
        else:
            dW_ep = W_ * xi_array[i + 1] - W_ep_start[i] * xi_array[i]
            cost_Base_pos = np.sum(np.where(dW_ep[dW_ep > 0], 1, 0))
            cost_Base_neg = np.sum(np.where(dW_ep[dW_ep < 0], 1, 0))
            cost = (
                215000
                * gas_price
                * 10 ** (-9)
                * price_set[z_moments_time[i + 1]]
                * (2 * cost_Base_pos + cost_Base_neg)
            )
        cost_ep[i + 1] = cost
        final_costs += cost

        W_ -= cost

        if W_ < 0:
            W_ = 0

        if ConstW:
            W_ = W_init

        W_ep_start[i + 1] = W_
        V_2d_ep = V_2d(
            price_set[z_moments_time[i + 1]],
            range_prices_func[2],
            V_init(mu_bucket, range_prices_func[1], range_prices_func[0]),
            1,
        )
        Li_ep = Li_Ei(
            price_set[z_moments_time[i + 1]],
            W_,
            xi_array[i + 1],
            ind_in_func,
            z_moments_time[i + 1],
            V_2d_ep,
        )
        V_3d_ep = V_3d(
            price_set[z_moments_time[i + 1] :],
            range_prices_func[2],
            V_init(mu_bucket, range_prices_func[1], range_prices_func[0]),
            Li_ep,
        )
        ret_ep = ret_AB(V_3d_ep, gamma, price_set[-1], fee_liq_dict, W_, i + 1, LIQ_fix, MCoef_adv)
        ret_ep_dW = ret_ep[1][0] * price_set[-1] + ret_ep[1][1]
        final_returns_ += ret_ep_dW

        W_end_ep = np.sum(V_3d_ep[-1, 1:-1, :2], axis=0)
        W_end_ep = W_end_ep[0] * price_set[-1] + W_end_ep[1]
        W_ = W_end_ep

        W_end_ep_bh = np.sum(V_3d_ep[0, 1:-1, :2], axis=0)
        bh_str[i + 1] = (W_end_ep_bh[0] * price_set[-1] + W_end_ep_bh[1]) / (
            W_end_ep_bh[0] * price_set[z_moments_time[i + 1]] + W_end_ep_bh[1]
        ) - 1

        if low_memory:
            pass
        else:
            ret_ep_st.append(
                ret_AB_(V_3d_ep, gamma, price_set[-1], fee_liq_dict, W_, i + 1, LIQ_fix, MCoef_adv)
            )

        if NO_cost:
            cost = 0
        else:
            cost = 215000 * gas_price * 10 ** (-9) * price_set[-1] * (2 * tau + 1)
        cost_ep[i + 2] = cost
        final_costs += cost
        W_ -= cost

        if W_ < 0:
            W_ = 0

        if ret_reloc:
            W_ += ret_ep_dW

        W_ep_start[i + 2] = W_
        fee_ep[i + 1] = ret_ep_dW

        if low_memory:
            V3_total = np.concatenate(
                (V3_total, np.stack([V_3d_ep[0], V_3d_ep[-1]], axis=0)), axis=0
            )
        else:
            V3_total = np.concatenate((V3_total, V_3d_ep), axis=0)

        Li_total = np.concatenate((Li_total, Li_ep), axis=0)
        Li_total = np.array(np.split(Li_total, t_ep, axis=0))
        ret_tab_total = np.concatenate((ret_tab_total, ret_ep[0]), axis=0)
        ret_tab_total = np.array((np.split(ret_tab_total, t_ep, axis=0)))

    else:

        i = 0
        W_ = W_init
        if NO_cost:
            cost = 0
        else:
            cost = (
                2 * 215000 * gas_price * 10 ** (-9) * price_set[z_moments_time[i]] * (2 * tau + 1)
            )
        W_ -= cost
        final_costs += cost
        cost_ep[i] = cost

        W_ep_start[i] = W_
        V_2d_ep = V_2d(
            price_set[z_moments_time[i]],
            range_prices_func[2],
            V_init(mu_bucket, range_prices_func[1], range_prices_func[0]),
            1,
        )
        Li_ep = Li_Ei(
            price_set[z_moments_time[i]], W_, xi_array[i], ind_in_func, z_moments_time[i], V_2d_ep
        )
        V_3d_ep = V_3d(
            price_set[z_moments_time[i] :],
            range_prices_func[2],
            V_init(mu_bucket, range_prices_func[1], range_prices_func[0]),
            Li_ep,
        )
        ret_ep = ret_AB(V_3d_ep, gamma, price_set[-1], fee_liq_dict, W_, i, LIQ_fix, MCoef_adv)
        ret_ep_st.append(
            ret_AB_(V_3d_ep, gamma, price_set[-1], fee_liq_dict, W_, i, LIQ_fix, MCoef_adv)
        )
        ret_ep_dW = ret_ep[1][0] * price_set[-1] + ret_ep[1][1]
        final_returns_ += ret_ep_dW

        W_end_ep = np.sum(V_3d_ep[-1, 1:-1, :2], axis=0)
        W_end_ep = W_end_ep[0] * price_set[-1] + W_end_ep[1]
        W_ = W_end_ep

        W_end_ep_bh = np.sum(V_3d_ep[0, 1:-1, :2], axis=0)
        bh_str[i] = (W_end_ep_bh[0] * price_set[-1] + W_end_ep_bh[1]) / (
            W_end_ep_bh[0] * price_set[z_moments_time[i]] + W_end_ep_bh[1]
        ) - 1

        if NO_cost:
            cost = 0
        else:
            cost = 215000 * gas_price * 10 ** (-9) * price_set[-1] * (2 * tau + 1)
        cost_ep[i + 1] = cost
        final_costs += cost
        W_ -= cost

        if W_ < 0:
            W_ = 0

        if ret_reloc:
            W_ += ret_ep_dW

        W_ep_start[i + 1] = W_
        fee_ep[i] = ret_ep_dW

        V3_total = V_3d_ep
        Li_total = Li_ep
        ret_tab_total = ret_ep[0]

    return (
        W_,
        W_ep_start,
        V3_total,
        Li_total,
        ret_tab_total,
        final_returns_,
        fee_ep,
        final_costs,
        cost_ep,
        ret_ep_st,
        bh_str,
    )


# let's write a function that determines the set of shares x_{i} for capital allocation -
# - depending on the reference bucket (including random set instead const parametr a)


def ratio_right(u_, ar_a):

    n_list = []
    k = 0

    for bucket_r in np.split(u_, (np.where(np.diff(u_) > 0)[0] + 1).tolist()):

        ar = ar_a[1 : len(bucket_r) + 1] / 2
        if k == 0:
            ar_T = ar
        else:
            ar_T = np.concatenate((ar_T, ar))
        k += 1

    return ar_T


def ratio_left(u_, ar_a):

    n_list = []
    k = 0

    for bucket_l in np.split(u_, (np.where(np.diff(u_) > 0)[0] + 1).tolist()):

        ar = np.flip(ar_a[1 : len(bucket_l) + 1]) / 2
        if k == 0:
            ar_T = ar
        else:
            ar_T = np.concatenate((ar_T, ar))
        k += 1

    return ar_T


def x_Nd_(tau, mu_bucket, p0, range_prices_, ar_a, flat_ratio=False):

    x_ind_Z = np.digitize(p0, range_prices_[2], right=True) - 1
    xi_arange = np.tile(np.arange(mu_bucket), (len(p0), 1))
    xi_ar = np.tile(np.zeros(mu_bucket), (len(p0), 1))

    if tau != 0:
        u_r = np.where(
            (xi_arange > x_ind_Z.reshape(-1, 1)) & (xi_arange < x_ind_Z.reshape(-1, 1) + tau + 1)
        )[0]
        xi_ar[
            np.where(
                (xi_arange > x_ind_Z.reshape(-1, 1))
                & (xi_arange < x_ind_Z.reshape(-1, 1) + tau + 1)
            )
        ] = ratio_right(u_r, ar_a)
        u_l = np.where(
            (xi_arange < x_ind_Z.reshape(-1, 1)) & (xi_arange > x_ind_Z.reshape(-1, 1) - tau - 1)
        )[0]
        xi_ar[
            np.where(
                (xi_arange < x_ind_Z.reshape(-1, 1))
                & (xi_arange > x_ind_Z.reshape(-1, 1) - tau - 1)
            )
        ] = ratio_left(u_l, ar_a)

    if flat_ratio:
        xi_ar[np.where(xi_arange == x_ind_Z.reshape(-1, 1))] = ar_a[0]
    else:
        xi_ar[np.where(xi_arange == x_ind_Z.reshape(-1, 1))] = 1 - np.sum(xi_ar, axis=1)

    return xi_ar, ar_a


# We take the data received using OHLC
eth_df = pd.read_csv(
    "/Users/gnapsamuel/Documents/AMM/Uniswap-v3-backtester-Source-code-and-scripts/data/eth_1m_upd.csv"
)
eth_df["time"] = pd.to_datetime(eth_df["time"])
eth_df = eth_df.set_index(["time"])

# let's select a series of prices with which we will work

price_test = eth_df["close"]
price_test = np.array(price_test)
swaps = pd.read_parquet(
    "/Users/gnapsamuel/Documents/AMM/proteus-LP-backtesting/data/uniswap/USDC-WETH/fee_500/swaps/swaps_2024-01-01_to_2024-01-10.parquet"
)
price_test = np.array(np.float64(swaps["final_price"]))

# select the mu parameter to satisfy the conditions
price_testty_ok = False

ty = 0.75
cu_pr = np.mean(prices).item()
W_initial = 100000
fact_return = np.sum(np.float64(swaps["dollar_value"])) * 0.0005
gas_pr = 100
tau = 100
gamma = 0.0005
price_test = prices
n_rg = 300
mu_bucket = n_rg
scen = 100
max_var = 10
min_var = 0.05
var_set = np.linspace(min_var, max_var, scen)
mu = ty
av_w = 11.5
prices_c = price_test.reshape(1, -1)
price_t_up = cu_pr + av_w * n_rg / 2
price_t_down = cu_pr - av_w * n_rg / 2
range_prices_ = range_prices(price_t_down, price_t_up, mu_bucket)

pr = prices_c[0]
ind_in = np.digitize(pr, range_prices_[2], right=True)
z_moments_t = z_list_t(tau, pr, ind_in)
t_ep = len(z_moments_t)
p0 = pr[z_moments_t]
var_ret = np.zeros((scen, 10))

k_r = 0
r_d = 3
x = np.linspace(-r_d, r_d, n_rg)

t = 20
var = var_set[t].item()
sigma = math.sqrt(var)
var, sigma
capacity = stats.norm.pdf(x, mu, sigma)
capacity /= np.sum(capacity)
xi_ar_ = np.tile(capacity, (t_ep, 1))
v3_m = v3_model(
    W_initial,
    False,
    gas_pr,
    t_ep,
    z_moments_t,
    pr,
    xi_ar_,
    range_prices_,
    ind_in,
    tau,
    low_memory=True,
    NO_cost=False,
    LIQ_fix=False,
)
