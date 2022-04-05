import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import mixture

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Categorical

class NormalMix:
    def __init__(self,pis,mus,covs):
        self.d = mus[0].shape[0]
        self.pis=torch.tensor(pis)
        self.mus = mus
        self.covs = covs
        self.mix = Categorical(torch.tensor(pis))
        self.comps = [MultivariateNormal(m[0], m[1]) for m in zip(mus,covs)] 

    def sample(self, n):
        cs=self.mix.sample(sample_shape=torch.Size([n]))
        Z=torch.zeros((n,self.d))
        for c in torch.unique(cs):
            Z[cs==c,:]=self.comps[c].sample(sample_shape=torch.Size([torch.sum(cs==c)]))
      
        return Z

    def log_prob(self, Z):
        like=torch.zeros(Z.shape[0])
        for c in range(len(self.comps)):
            like+=self.pis[c]*torch.exp(self.comps[c].log_prob(Z))

        return torch.log(like)

class NormalMixFit:
    def __init__(self, num_comps=[2,3,4,5,6], criterion='bic', reg_covar=1e-06, random_state=42):
        self.num_comps = num_comps
        self.criterion = criterion
        self.reg_covar = reg_covar
        self.random_state = random_state

    def gmm_scores(self, Z, k):
        clf = mixture.GaussianMixture(n_components=k, covariance_type="full", reg_covar=self.reg_covar, random_state=self.random_state)
        clf.fit(Z)
        if self.criterion=='aic':
            return clf.aic(Z)
        elif self.criterion=='bic':
            return clf.bic(Z)

    def get_gmm(self, X, y):
        Z=np.hstack((X, y.reshape((-1,1))))
        scores=np.array([self.gmm_scores(Z, k) for k in self.num_comps])
        k_star=self.num_comps[np.argmin(scores)]

        #Training GMM
        Z = np.hstack((y.reshape((-1,1)),X))
        gmm = mixture.GaussianMixture(n_components=k_star, covariance_type="full", reg_covar=self.reg_covar, random_state=self.random_state)
        gmm.fit(Z)
        self.fitted_gmm = gmm
    
    
class CayleyArmijo:
    def __init__(self, rho_1, rho_2, max_iter):
        self.rho_1 = rho_1
        self.rho_2 = rho_2
        self.max_iter = max_iter

    def get_params(self):
        return [self.rho_1, self.rho_2, self.max_iter]

class SDR:
    def __init__(self, p, eta, epochs, n=1000, n_lb=100000, stoc=True, algo='natgd', early_stopping=10):
        self.n = n
        self.n_lb = n_lb
        self.p = p
        self.eta = eta
        self.epochs = epochs
        self.stoc = stoc
        self.algo = algo
        self.early_stopping = early_stopping

    def fit(self, fitted_gmm):
        self.pis = fitted_gmm.weights_.tolist()
        self.mus = [torch.tensor(x) for x in fitted_gmm.means_.tolist()]
        self.covs = [torch.tensor(x) for x in fitted_gmm.covariances_.tolist()]

        # get LB
        q=self.mus[0].shape[0]-1 ##put condition on p
        Ex = torch.cat([torch.eye(q), torch.zeros(q,1)], dim=1).requires_grad_(False)
        Ey = torch.zeros((1,q+1)).requires_grad_(False)
        Ey[0,q]=1

        n_comp = len(self.pis)
        zdist = NormalMix(self.pis,self.mus,self.covs)
        Z=zdist.sample(self.n_lb)

        ymus=[Ey@self.mus[i] for i in range(n_comp)]
        ycovs=[Ey@self.covs[i]@Ey.T for i in range(n_comp)]
        ydist = NormalMix(self.pis,ymus,ycovs)

        xmus=[Ex@self.mus[i] for i in range(n_comp)]
        xcovs=[Ex@self.covs[i]@Ex.T for i in range(n_comp)]
        xdist = NormalMix(self.pis,xmus,xcovs)

        self.lb = -float((zdist.log_prob(Z)-xdist.log_prob(Z@Ex.T)-ydist.log_prob(Z@Ey.T)).mean())

        # find optimum A
        Z=zdist.sample(self.n)

        ymus=[Ey@self.mus[i] for i in range(n_comp)]
        ycovs=[Ey@self.covs[i]@Ey.T for i in range(n_comp)]
        ydist = NormalMix(self.pis,ymus,ycovs)

        S=np.cov(Z[:,:q].T)
        V,U=np.linalg.eig(S)
        A = torch.tensor(U[:,:self.p].T).float().requires_grad_(True)  
        B = torch.cat([A, torch.zeros(self.p,1)], dim=1) 
        B = torch.cat([B,torch.cat([torch.zeros(1,q),torch.ones(1,1)], dim=1)], dim=0)

        kmus=[A@Ex@self.mus[i] for i in range(n_comp)]
        kcovs=[A@Ex@self.covs[i]@Ex.T@A.T for i in range(n_comp)]
        kdist = NormalMix(self.pis,kmus,kcovs)
        kymus=[B@self.mus[i] for i in range(n_comp)]
        kycovs=[B@self.covs[i]@B.T for i in range(n_comp)]
        kydist = NormalMix(self.pis,kymus,kycovs)

        loss_hist=[]
        best_val_loss, best_val_epoch = None, None

        for j in tqdm(range(self.epochs)):

            loss=-(kydist.log_prob(Z@B.T)-kdist.log_prob(Z@(A@Ex).T)-ydist.log_prob((Z@(Ey).T))).mean() 
            loss_hist.append(loss.item())
            if type(self.early_stopping) == int:
                if best_val_loss == None or best_val_loss > loss_hist[-1]:
                    best_val_loss, best_val_epoch = loss_hist[-1], j
                if best_val_epoch < j - self.early_stopping:
                    break
            loss.backward(retain_graph=True)
            G=torch.clone(A.grad).T
            A.grad.zero_()

            ### Updating A ###

            # Method 1 (natgd w/ eta fixed)
            if self.algo == "natgd":
                A = A @ torch.matrix_exp(-self.eta*(A.T@G.T-G@A))

            # Method 2 (Cayley w/ eta fixed)
            elif self.algo == "cayley":
                if q < 2 * self.p:
                    W = G @ A - A.T @ G.T
                    A = A @ (torch.eye(q) + self.eta/2 * W) @ torch.inverse(torch.eye(q) - self.eta/2 * W)
                else: #use S-M-W formula for speedup
                    U = torch.cat([G, A.T], dim = 1)
                    V = torch.cat([A.T, -1 * G], dim = 1)
                    A = A - self.eta * A @ V @ torch.inverse(torch.eye(2*self.p) + self.eta/2 * U.T @ V) @ U.T

             # Method 3 (Cayley w/ Armijo's rule)
            elif type(self.algo) == CayleyArmijo:
                rho_1, rho_2, max_iter = self.algo.get_params()
                orig_loss = loss
                W = G @ A - A.T @ G.T
                U = torch.cat([G, A.T], dim = 1)
                V = torch.cat([A.T, -1 * G], dim = 1)
                der_0 = torch.trace(A @ W @ G)
                cntr = 0
                eta_init = self.eta
                while True:
                    if cntr == max_iter: # to avoid infinite loop, stop the algorithm once stopping condition is not reached after 10 iterations
                        break
                    if q < 2 * self.p:
                        inv_temp = torch.inverse(torch.eye(q) - eta_init/2 * W)
                    else:
                        inv_temp = torch.eye(q) - eta_init/2 * V @ torch.inverse(torch.eye(2*self.p) + eta_init/2 * U.T @ V) @ U.T
                    A_cand = A @ (torch.eye(q) + eta_init/2 * W) @ inv_temp
                    B_cand = torch.cat([A_cand, torch.zeros(self.p,1)], dim=1) 
                    B_cand = torch.cat([B_cand,torch.cat([torch.zeros(1,q),torch.ones(1,1)], dim=1)], dim=0)
                    kmus=[A_cand@Ex@self.mus[i] for i in range(n_comp)]
                    kcovs=[A_cand@Ex@self.covs[i]@Ex.T@A_cand.T for i in range(n_comp)]
                    kdist = NormalMix(self.pis,kmus,kcovs)
                    kymus=[B_cand@self.mus[i] for i in range(n_comp)]
                    kycovs=[B_cand@self.covs[i]@B_cand.T for i in range(n_comp)]
                    kydist = NormalMix(self.pis,kymus,kycovs)
                    cand_loss=-(kydist.log_prob(Z@B_cand.T)-kdist.log_prob(Z@(A_cand@Ex).T)-ydist.log_prob((Z@(Ey).T))).mean().item()
                    der_eta = 0.5 * torch.trace((A + A_cand) @ W @ inv_temp @ G)
                    if cand_loss <= orig_loss + rho_1 * eta_init * der_0 and der_eta >= rho_2 * der_0:
                        A = A_cand
                        break
                    else:
                        eta_init *= 0.5
                        cntr += 1

              ### Preparing for the next round ###
            A = A.clone().detach().requires_grad_(True) 
            B = torch.cat([A, torch.zeros(self.p,1)], dim=1) 
            B = torch.cat([B,torch.cat([torch.zeros(1,q),torch.ones(1,1)], dim=1)], dim=0)

            kmus=[A@Ex@self.mus[i] for i in range(n_comp)]
            kcovs=[A@Ex@self.covs[i]@Ex.T@A.T for i in range(n_comp)]
            kdist = NormalMix(self.pis,kmus,kcovs)
            kymus=[B@self.mus[i] for i in range(n_comp)]
            kycovs=[B@self.covs[i]@B.T for i in range(n_comp)]
            kydist = NormalMix(self.pis,kymus,kycovs)

            if self.stoc:
                Z=zdist.sample(self.n)

        ### Storing ###
        self.A = A
        self.loss_hist = loss_hist

    def plot(self):
        plt.plot(self.loss_hist,alpha=.5);
        plt.axhline(y=self.lb, color='b', linestyle='--')

        
# p chosen such that the loss / LB is > some threshold; use binary search to accelerate from O(N) to O(LOG N)
class DynamicSDR:
    def __init__(self, eta, epochs, n=1000, n_lb=100000, stoc=True, algo='natgd', early_stopping=10, loss_ratio=0.5):
        self.n = n
        self.n_lb = n_lb
        self.eta = eta
        self.epochs = epochs
        self.stoc = stoc
        self.algo = algo
        self.early_stopping = early_stopping
        self.loss_ratio = loss_ratio

    def fit(self, fitted_gmm):
        self.mus = [torch.tensor(x) for x in fitted_gmm.means_.tolist()]
        low = 0
        high = self.mus[0].shape[0]-1
        arr_res = [None] * high
        while low != high:
            mid = int((low + high)/2)
            sdr_model = SDR(n=self.n, n_lb=self.n_lb, p=mid+1, eta=self.eta, epochs=self.epochs, stoc=self.stoc, algo=self.algo, 
                            early_stopping=self.early_stopping)
            sdr_model.fit(fitted_gmm)
            arr_res[mid] = sdr_model
            if sdr_model.loss_hist[-1]/sdr_model.lb <= self.loss_ratio:
                low = mid + 1
            else:
                high = mid
        self.A = arr_res[low].A
        self.loss_hist = arr_res[low].loss_hist
        self.lb = arr_res[low].lb

    def plot(self):
        plt.plot(self.loss_hist,alpha=.5);
        plt.axhline(y=self.lb, color='b', linestyle='--')
