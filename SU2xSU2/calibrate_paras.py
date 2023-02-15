import numpy as np
from SU2xSU2 import SU2xSU2


def calibrate(model_paras, sim_paras=None, production_run=False, accel=False):
    '''For a model, specified by the dictionary model_paras, this function calibrates the values of ell and eps to produce an acceptance rate in the desireable range between 60 and 80%.
    When acceptance rate is outside this range, the number of steps is adjusted according to the difference to the ideal acceptance rate of 65%. The step size if fixed by requiring
    trajectories to be of unit length. To avoid getting caught in a loop, the calibration is limited to 10 iterations.
    When sim_paras is not passed, 500 trajectories with no thinning and 50% burn in are simulated to ensure the calibration process is fast. These parameters can be 
    overwritten by passing an appropriate dictionary. 
    When wanting to fine tune the parameters for a production run, set production_run=True and specify the simulation further by passing sim_paras. This will also return the SU2xSU2 instance
    It is advise to perform the above described rough calibration beforehand.
    
    model_paras: dict
        {N, a, ell, eps, beta} with ell, eps as guesses to start the calibration. Their product must be 1
    sim_paras: dict
        {M, thin_freq, burnin_frac, 'renorm_freq':10000, accel, 'store_data':False}
    production_run: bool
        set to True to return the calibrated SU2xSU2 instance 

    Returns
    if not production run:
    model_paras: dict
        calibrated model parameters
    else:
    model: SU2xSU2 object
        result of calibrated simulation to take measurements later on
    model_paras: as above
    '''
    # defining bounds for desireable acceptance rate
    lower_acc, upper_acc = 0.55, 0.8

    if sim_paras is None:
        # default for fast calibration
        sim_paras = {'M':500, 'thin_freq':1, 'burnin_frac':0.5, 'renorm_freq':10000, 'accel':accel, 'store_data':False}
        # use narrower range for desired acceptance to avoid barely passing fast calibration and then not passing during the production run, causing a much longer simulation to be repeated
        lower_acc, upper_acc = 0.6, 0.75
    
    good_acc_rate = False
    count = 0 
    while good_acc_rate == False:
        model = SU2xSU2(**model_paras)
        model.run_HMC(**sim_paras)  
        acc_rate = model.acc_rate
        d_acc_rate = 0.65 - acc_rate
        if count >= 10:
            good_acc_rate = True
        if acc_rate < lower_acc or acc_rate > upper_acc:
            new_ell = int(np.rint(model_paras['ell']*(1 + d_acc_rate)))
            # due to rounding it can happen that ell is not updated. To avoid getting stuck in a loop, enforce minimal update of +/- 1
            if new_ell == model_paras['ell']:
                if d_acc_rate > 0:
                    new_ell += 1
                else:
                    new_ell -= 1
                    if new_ell == 0:
                        break # stop calibration when step size has to be reduce below 1. 
            model_paras['ell'] = new_ell
            model_paras['eps'] = 1/model_paras['ell']
            count +=1
        else:
            good_acc_rate = True

    if production_run:
        return model, model_paras

    return model_paras