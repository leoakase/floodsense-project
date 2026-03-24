import numpy as np
np.random.seed(15)

#monte carlo simulation of a probabilistic flood prediction tool

def scenario():
    "can we vectorize?"
    rainfall=np.random.uniform(20, 150)
    infiltration=np.random.uniform(10, 35)
    drainage_capacity=np.random.uniform(15, 60)
    runoff_coefficient=np.random.uniform(0.4, 0.9)
    slope=np.random.uniform(0, 10)
    return rainfall, infiltration, drainage_capacity, runoff_coefficient, slope

def lognormal_params(mean, std):
    var = std**2
    sigma = np.sqrt(np.log(1 + var / mean**2))
    mu = np.log(mean**2 / np.sqrt(var + mean**2))
    return mu, sigma


def monte_carlo(rain, infi, drain, runoff, slope, M, shape=4, infi_std=3, drain_std=5, runoff_std=0.05, slope_std=1):

    """
    function to implement vectorized form of the monte carlo simulatio with some default arguement values

    """

    scale=rain/shape

    nrain=np.random.gamma(shape, scale, M)
    mu, sigma=lognormal_params(infi, infi_std) #mu and sigma for infi
    ninfi=np.random.lognormal(mu, sigma, M)
    mu, sigma=lognormal_params(drain, drain_std) #mu and sigma rebinded to be for drain
    ndrain=np.random.lognormal(mu, sigma, M)
    nrunoffcoe=np.random.normal(runoff, runoff_std, M)
    nslope=np.random.normal(slope, slope_std, M)
    #in place arithmetic to avoid temp arrays for runoff_val=nrain*(nrunoffcoe*(1 + 0.05*nslope)) to use in flood 
    #condition if runoff_val > (ninfi + ndrain) then flood occurs
    nslope*=0.05
    nrunoffcoe*=(1 + nslope)
    nrain*=nrunoffcoe
    runoff_val=nrain

    bound=ninfi + ndrain
    prob=np.mean(runoff_val > bound)

    return prob


def generate_dataset(N, M):

    "can we vectorize? any need for that?"


    data=[]

    for k in range(N):
        rain, infi, drain, runoff, slope=scenario()
        prob=monte_carlo(rain, infi, drain, runoff, slope, M)
        data.append([rain, infi, drain, runoff, slope, prob])
    return data


if __name__== "__main__":
    pass