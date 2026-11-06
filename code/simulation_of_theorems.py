import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit
from sklearn.metrics import *
from datetime import datetime
from scipy.stats import spearmanr
from pathlib import Path
from matplotlib.pyplot import text
import matplotlib.ticker as ticker

class Simulation():
    def __init__(self, plausibility_function,
                 human_factor_function,
                 gt = None,
                 ai_pred = None,
                 ai_pred_prob = None,
                 human_pred = None,
                 perf_metric = accuracy_score,
                 num_class = 5,
                 num_samples = 1000,
                 human_sim_acc = 0.8,
                 ai_sim_acc =0.9,
                 **kwargs):
        '''
        Generate a simulation experiment data. One instance is a simulated trial.
        For a multiclass classification problem, generate pseudo human-subject experiment result table, returned as study_result.
        The result table will be used to analyzing the relationship between plausibility and complementary human-ai team performance.
        '''
        self.verifiability = None  # explanation goal of verifying AI decisions, correlation between gt prob and plausibility
        self.human_perf = None
        self.ai_perf = None
        self.team_perf = None
        self.complement = None # bool variable to indicate complementary is achieved or not
        self.E_fw = None
        self.E_fr = None
        self.E_f_slope = None
        self.E_f_intercept = None
        self.theorem2_fulfilled = None
        self.__dict__.update(kwargs)
        self.perf_metric = perf_metric
        self.classes = list(range(1, num_class+1))
        self.ai_sim_acc = ai_sim_acc
        self.human_sim_acc = human_sim_acc
        self.num_samples = num_samples
        self.num_class = num_class
        self.human_acceptance = list() # r.v. B in the paper
        if (ai_pred is not None) and (ai_pred_prob is not None) and (human_pred is not None) and (gt is not None):
            self.ai_pred = ai_pred
            self.ai_pred_prob = ai_pred_prob
            self.human_pred = human_pred
            self.gt = gt
            self.num_samples = len(self.gt)
        else:
            # gt column
            self.gt = np.random.choice(self.classes, size=num_samples)
            # ai_pred and human_pred columns
            self.ai_pred = list()
            self.ai_pred_prob = list() # the probability of the gt class
            self.human_pred = list()
            self.num_samples = num_samples
            for i in range(self.num_samples):
                # ai_pred: in the multiclassification problem, the ai_pred is the gt probability. if ai_pred > 0.5, then it is correct.
                if np.random.uniform() < self.ai_sim_acc:
                    self.ai_pred_prob.append(np.random.uniform(0.5, 1))
                    self.ai_pred.append(self.gt[i])
                else:
                    self.ai_pred_prob.append(np.random.uniform(0, 0.5))
                    copied_classes = list(self.classes)
                    copied_classes.remove(self.gt[i])
                    non_gt = np.random.choice(copied_classes)
                    self.ai_pred.append(non_gt)
                # human_pred
                if np.random.uniform() < self.human_sim_acc:
                    self.human_pred.append(self.gt[i])
                else:
                    copied_classes = list(self.classes)
                    copied_classes.remove(self.gt[i])
                    non_gt = np.random.choice(copied_classes)
                    self.human_pred.append(non_gt)
        # plausibility in the range (0, 1)
        self.plausibility = plausibility_function(**self.__dict__)
        # f_p is f(P), the probability that human will accept an AI decision
        self.f_p = human_factor_function(**self.__dict__)
        # team_pred
        self.team_pred = list()
        for i in range(self.num_samples):
            if np.random.uniform() < self.f_p[i]:
                self.team_pred.append(self.ai_pred[i])
                self.human_acceptance.append(True)
            else:
                self.team_pred.append(self.human_pred[i])
                self.human_acceptance.append(False)
        self.data_dict = {'gt': self.gt, 'AI_pred': self.ai_pred, 'AI_pred_prob': self.ai_pred_prob, 'human_pred': self.human_pred, 'team_pred': self.team_pred, 'plausibility': self.plausibility, 'f_p': self.f_p, 'human_acceptance': self.human_acceptance}
        self.study_result = pd.DataFrame.from_dict(self.data_dict)
        self.study_result['AI_correctness'] = self.study_result['AI_pred'] == self.study_result['gt'] # # r.v. C in the paper


    def get_trial_data(self):
        return self.study_result


    def get_performance(self, to_print = False):
        self.human_perf = self.perf_metric(self.study_result['gt'], self.study_result['human_pred'])
        self.ai_perf = self.perf_metric(self.study_result['gt'], self.study_result['AI_pred'])
        self.team_perf = self.perf_metric(self.study_result['gt'], self.study_result['team_pred'])
        self.complement = self.team_perf > max(self.human_perf, self.ai_perf)
        self.verifiability = spearmanr(self.ai_pred_prob, self.plausibility)[0]
        if to_print:
            print("Human performance h is: {}, AI performance h is {}, team performance t is {}. Complementary performance achieved: {}".format(self.human_perf , self.ai_perf , self.team_perf , self.complement))
        return self.human_perf, self.ai_perf, self.team_perf, self.complement

    def get_Ef(self, to_print = False):
        self.E_fr =  self.study_result[(self.study_result['AI_correctness'] == True) & (self.study_result['human_acceptance'] == True)].shape[0]/self.study_result[self.study_result['AI_correctness'] == True].shape[0]
        self.E_fw = self.study_result[(self.study_result['AI_correctness'] == False) & (self.study_result['human_acceptance'] == True)].shape[0]/self.study_result[self.study_result['AI_correctness'] == False].shape[0]
        m_1_minus_h = self.ai_perf * (1- self.human_perf)
        h_1_minus_m = self.human_perf * (1-self.ai_perf)
        self.E_f_slope = h_1_minus_m/ m_1_minus_h
        intercept = (self.ai_perf - self.human_perf)/m_1_minus_h
        if self.human_perf >= self.ai_perf:
            self.E_f_intercept = 0
        else:
            self.E_f_intercept = intercept
        self.theorem2_fulfilled = self.E_fr > self.E_f_slope * self.E_fw + self.E_f_intercept

        if to_print:
            print("E_fr is {}, E_fw is {}. Theorem 2 fulfilled: {}".format(self.E_fr, self.E_fw, self.theorem2_fulfilled))
            print('---Theorem2 is verified to be {}---'.format(self.complement == self.theorem2_fulfilled))
        return self.E_fr, self.E_fw, self.theorem2_fulfilled, self.E_f_slope, self.E_f_intercept

    def get_all_metrics(self):
        self.get_performance()
        self.get_Ef()
        metric_dict = {"E_fr": self.E_fr, "E_fw": self.E_fw,
                       "E_f_slope": self.E_f_slope, "E_f_intercept": self.E_f_intercept,
                       "verifiability": self.verifiability,
                       "theorem2_fulfilled": self.theorem2_fulfilled, "human_perf": self.human_perf, "ai_perf": self.ai_perf, "team_perf": self.team_perf, "complement": self.complement}
        return metric_dict


class PlotTrials:
    def __init__(self,
                 plausibility_function = None,
                 human_factor_function = None,
                 fixed_h_m = True,
                 num_trials=1000,
                 perf_metric = accuracy_score,
                 num_class = 10,
                 num_samples = 2000,
                 human_sim_acc = 0.9,
                 ai_sim_acc =0.8,
                 **kwargs):
        self.num_trials = num_trials
        self.fixed_h_m = fixed_h_m # the human and AI performance is fixed
        self.trial_data = None
        self.plausibility_function = plausibility_function
        self.human_factor_function = human_factor_function
        self.perf_metric = perf_metric
        self.num_class = num_class
        self.num_samples = num_samples
        self.human_sim_acc = human_sim_acc
        self.ai_sim_acc = ai_sim_acc
        self.__dict__.update(kwargs)
        self.gt = None
        self.human_pred = None
        self.ai_pred = None
        self.ai_pred_prob = None
        self.df = None
        if self.fixed_h_m:
            trial = Simulation(plausibility_function=plau_gaussian,
                               human_factor_function=fp_linear,
                               num_class=self.num_class,
                               num_samples=self.num_samples,
                               human_sim_acc=self.human_sim_acc,
                               ai_sim_acc=self.ai_sim_acc,
                               weight=1, bias=0, upper=1, lower=0.8, mu=0.8, sigma=0.1)
            trial_data = trial.get_trial_data()
            self.gt = trial_data['gt']
            self.human_pred = trial_data['human_pred']
            self.ai_pred = trial_data['AI_pred']
            self.ai_pred_prob = trial_data['AI_pred_prob']
            self.human_perf, self.ai_perf, _, _ = trial.get_performance()
        # self.kwargs = self.__dict__


    def run_trials(self):
        if self.fixed_h_m:
            assert  (self.ai_pred is not None) and (self.ai_pred_prob is not None) and (self.human_pred is not None) and (self.gt is not None)
        trial_data_list = list()
        for i in range(self.num_trials):
            if self.plausibility_function is None:
                plausibility_generator = np.random.choice([plau_uniform, plau_gaussian,
                                                           plau_corr_uniform, plau_corr_gaussian, plau_corr_prob])
            else:
                plausibility_generator = self.plausibility_function
            if self.human_factor_function is None:
                fp_generator = np.random.choice([fp_sigmoid, fp_linear])
            else:
                fp_generator = self.human_factor_function
            trial = Simulation(plausibility_function=plausibility_generator,
                               human_factor_function=fp_generator,
                               num_class=self.num_class,
                               num_samples=self.num_samples,
                               human_acc=self.human_sim_acc,
                               ai_acc=self.ai_sim_acc,
                               ai_pred = self.ai_pred,
                               ai_pred_prob = self.ai_pred_prob,
                               human_pred= self.human_pred,
                               gt = self.gt,
                               weight= np.random.uniform(0.8, 1.2), # parameters for fp_linear
                               bias= np.random.uniform(0, 0.3),
                               upper=np.random.uniform(0.6, 1),  # parameters for plau_uniform
                               lower=np.random.uniform(0.0, 0.4),
                               mu=np.random.uniform(0.1, 0.9), # parameters for plau_gaussian
                               sigma=np.random.uniform(0.01, 0.5),
                               upper_r = np.random.uniform(1,3), # parameters for plau_corr_uniform
                               lower_r = np.random.uniform(0.5, 1),
                               upper_w = np.random.uniform(-0.9, 0.5),
                               lower_w = np.random.uniform(-3, -1),
                               mu_r = np.random.uniform(0,3),  # parameters for plau_corr_gaussian
                               sigma_r = np.random.uniform(0.5, 5),
                               mu_w = np.random.uniform(-3, 0),
                               sigma_w = np.random.uniform(0.5, 5),
                               )
            ## todo record the experiment parameters
            trial_metric = trial.get_all_metrics()
            # add experiment parameter to trial_metric
            # if self.fixed_h_m:
            #     trial_metric['human_perf'] = self.human_pred
            trial_data_list.append(pd.DataFrame([trial_metric]))
        self.df = pd.concat(trial_data_list)
        if self.fixed_h_m:
            assert len(self.df['human_perf'].unique())==1 and len(self.df['ai_perf'].unique())==1
        return self.df

    def plot(self):
        # markers = {True: "o", False: "*"}
        self.run_trials()
        self.df = self.df.sort_values(by=['complement'])
        sns.set_theme(rc={'figure.figsize': (5, 5)})
        with sns.plotting_context("talk", font_scale=1):
            sns.set_style("whitegrid")
            axes = plt.gca()
            # add line to show the relationship of theorem 2
            if self.fixed_h_m:
                assert len(self.df['E_f_slope'].unique()) == 1 and len(self.df['E_f_intercept'].unique()) == 1
                x_vals = np.array(axes.get_xlim())
                y_vals = self.df['E_f_intercept'].unique()[0] + self.df['E_f_slope'].unique()[0] * x_vals
                plt.plot(x_vals, y_vals, 'r-', zorder=0)
            g1 = sns.scatterplot(
                data=self.df, x='E_fw', y='E_fr',  hue= 'complement', hue_order = [False, True],
                legend = False,
                s=20
                # size = 'verifiability',
                # size="theorem2_fulfilled",
                # markers=markers,  sizes = {False: 50, True:500}
                )
            g1.set(xlim=(0, 1), ylim=(0, 1), xticks=[0, .5, 1], yticks=[0, .5, 1])
            plt.xlabel(r"$\mathbb{E}[{f^w}]$")
            plt.ylabel(r"$\mathbb{E}[{f^r}]$")
            plt.title("Human acc: {:.2f}, AI acc: {:.2f}".format(self.human_perf, self.ai_perf))

            # g1.legend(loc='center left', bbox_to_anchor=(1, 0.5), labels=["Theorem 2",'No complementarity', 'Complementarity'])

            dateTimeObj = datetime.now()
            time_stamp = dateTimeObj.strftime("%Y%m%d_%H%M")
            fig_name = 'human_{}-ai_{}-{}.pdf'.format(self.human_perf, self.ai_perf, time_stamp)
            plt.savefig(Path('../result')/ fig_name, bbox_inches="tight")
        return self.df



def plau_uniform(**kwargs):
    # plausibility_function: no correlation with decision correctness, uniform distribution in (0,1)
    if 'upper' in kwargs:
        upper = kwargs['upper']
        upper = np.min([upper, 1])
    else:
        upper = 1
    if 'lower' in kwargs:
        lower = kwargs['lower']
        lower = np.max([lower, 0])
    else:
        lower = 0
    return np.random.uniform(low = lower, high = upper, size = kwargs['num_samples'])

def plau_gaussian(**kwargs):
    # plausibility_function: no correlation with decision correctness, gaussian distribution clampped in (0,1)
    if 'mu' not in kwargs:
        raise ValueError("Please provide value for mu")
    if 'sigma' not in kwargs:
        raise ValueError("Please provide value for sigma")
    mu = kwargs['mu']
    sigma = kwargs['sigma']
    gaussian = np.random.normal(mu, sigma, size = kwargs['num_samples'])
    # gaussian[gaussian < 0] = 0
    # gaussian[gaussian > 1] = 1
    return gaussian

def plau_corr_uniform(**kwargs):
    # plausibility_function: correlation with decision correctness, uniform distribution of right/wrong predictions
    if 'gt' not in kwargs:
        raise ValueError("Please provide value for gt")
    if 'ai_pred' not in kwargs:
        raise ValueError("Please provide value for ai_pred")
    if 'upper_r' not in kwargs:
        raise ValueError("Please provide value for upper_r")
    if 'upper_w' not in kwargs:
        raise ValueError("Please provide value for upper_w")
    if 'lower_r' not in kwargs:
        raise ValueError("Please provide value for lower_r")
    if 'lower_w' not in kwargs:
        raise ValueError("Please provide value for lower_w")
    gt = kwargs['gt']
    ai_pred = kwargs['ai_pred']
    plausibility = np.zeros(len(gt))
    plausibility[ai_pred == gt] = np.random.uniform(low=kwargs['lower_r'], high=kwargs['upper_r'], size = len(plausibility[ai_pred == gt]))
    plausibility[ai_pred != gt] = np.random.uniform(low=kwargs['lower_w'], high=kwargs['upper_w'], size = len(plausibility[ai_pred != gt]))
    return plausibility

def plau_corr_gaussian(**kwargs):
    # plausibility_function in (0,1): correlation with decision correctness, gaussian distribution of right/wrong predictions
    if 'gt' not in kwargs:
        raise ValueError("Please provide value for gt")
    if 'ai_pred' not in kwargs:
        raise ValueError("Please provide value for ai_pred")
    if 'mu_r' not in kwargs:
        raise ValueError("Please provide value for mu_r")
    if 'mu_w' not in kwargs:
        raise ValueError("Please provide value for mu_w")
    if 'sigma_r' not in kwargs:
        raise ValueError("Please provide value for sigma_r")
    if 'sigma_w' not in kwargs:
        raise ValueError("Please provide value for sigma_w")
    gt = kwargs['gt']
    ai_pred = kwargs['ai_pred']
    plausibility = np.zeros(len(gt))
    plausibility[ai_pred == gt] = np.random.normal(loc=kwargs['mu_r'], scale=kwargs['sigma_r'], size = len(plausibility[ai_pred == gt]) )
    plausibility[ai_pred != gt] = np.random.normal(loc=kwargs['mu_w'], scale=kwargs['sigma_w'], size = len(plausibility[ai_pred != gt]) )
    return plausibility

def plau_corr_prob(**kwargs):
    # plausibility_function in (0,1): correlation with gt probability
    if 'ai_pred_prob' not in kwargs:
        raise ValueError("Please provide value for ai_pred_prob")
    return kwargs['ai_pred_prob'] + np.random.normal(0, 5, len(kwargs['ai_pred_prob']))


def fp_sigmoid(**kwargs):
    # human_factor_function in (0,1): non-linear monotonically non-decreasing function
    if 'plausibility' not in kwargs:
        raise ValueError("Please provide value for plausibility")
    return expit(kwargs['plausibility'])

def fp_linear(**kwargs):
    # human_factor_function in (0,1):  linear function
    if 'weight' not in kwargs:
        raise ValueError("Please provide value for weight")
    if 'bias' not in kwargs:
        raise ValueError("Please provide value for bias")
    plausibility = kwargs['plausibility']
    weight = kwargs['weight']
    bias = kwargs['bias']
    f_p = plausibility * weight + bias + np.random.normal(0, 1, len(kwargs['ai_pred_prob']))
    # clam to [0,1]
    # f_p[f_p < 0] = 0
    # f_p[f_p > 1] = 1
    return f_p


def plot_h_m_relationship(acc_list = [0.6, 0.7, 0.8, 0.9], figsize= 5):
    # color_grad_dict = {0.1: '#2c7bb6', 0.2: '#fdae61', 0.3: '#d7191c'}
    if np.max(acc_list) >0.8:
        color_grad_dict = {(0.6, 0.7): '#91DBE1', (0.7, 0.8): '#6990E0', (0.8, 0.9): '#0300E1',
                           (0.6, 0.8): '#C0EB7A',  (0.7, 0.9): '#00EA23',
                           (0.6, 0.9): '#DC0B04'
                           }
    else:
        color_grad_dict = {(0.2, 0.3): '#91DBE1', (0.3, 0.4): '#6990E0', (0.4, 0.5): '#0300E1',
                           (0.2, 0.4): '#C0EB7A',  (0.3, 0.5): '#00EA23',
                           (0.2, 0.5): '#DC0B04'
                           }

    # plot to show the relationship of h and m in theorem2
    sns.set_theme(rc={'figure.figsize': (figsize, figsize)})
    with sns.plotting_context("talk", font_scale=1):
        sns.set_style("whitegrid")

        axes = plt.gca()
        axes.set(xlim=(0, 1), ylim=(0, 1), xticks=[0, .5, 1], yticks=[0, .5, 1])

        # add line to show the relationship of theorem 2
        for h in acc_list:
            for m in acc_list:
                x_vals = np.array(axes.get_xlim())
                E_f_slope = h * (1-m) / (m*(1-h))
                if h<m:
                    E_f_intercept = (m-h)/(m*(1-h))
                else:
                    E_f_intercept = 0
                y_vals =  E_f_slope * x_vals + E_f_intercept
                if h<m:
                    plt.plot(x_vals, y_vals, color = color_grad_dict[(h, m)], label =  'h={:.1f}, m={:.1f}'.format(h, m))
                    plt.annotate( 'h={:.1f}, m={:.1f}'.format(h, m).replace("0", ""),
                                xy=(0, y_vals[0]),
                                xytext=(-0.01, y_vals[0]),
                                color=color_grad_dict[(h, m)],
                                horizontalalignment='right',
                                verticalalignment="center"
                                )
                else:
                    if h == m:
                        plt.plot(x_vals, y_vals, color = 'gray')
                    else:
                        plt.plot(x_vals, y_vals, color = color_grad_dict[(m, h)], linestyle='dashed', label =  'h={:.1f}, m={:.1f}'.format(h, m))
                        plt.annotate('h={:.1f}, m={:.1f}'.format(h, m).replace("0", ""),
                                     xy=(1/E_f_slope, 1),
                                     xytext=(1/E_f_slope, 1.02),
                                     color=color_grad_dict[(m, h)],
                                     rotation=90,
                                     horizontalalignment='center',
                                     verticalalignment="bottom"
                                     )
        text(0.5, 0.5, "h=m", rotation=45, verticalalignment='center', color = 'gray')
        # plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
        plt.xlabel(r"$\mathbb{E}[{f^w}]$")
        plt.ylabel(r"$\mathbb{E}[{f^r}]$")
        axes.yaxis.set_label_position("right")
        axes.yaxis.tick_right()

        dateTimeObj = datetime.now()
        time_stamp = dateTimeObj.strftime("%Y%m%d_%H%M")
        fig_name = '{}-{}.pdf'.format('h-and-m', time_stamp)
        plt.savefig(Path('../result')/ fig_name, bbox_inches="tight")

def plot_h_m_heatmap(figsize= 5):
    '''a heatmap with h, m taking different values at the x-y axis, the color of heatmap encode the area above the E_fr line showing the probability of achieving complementarity'''
    # construct the heatmap dataset
    heatmap_dict = dict()
    nums = np.linspace(0.001, 0.999, 101)
    for h in nums:
        heatmap_col = dict()
        for m in nums:
            if h < m:
                area = h*(1 - m) / (2* m * (1 - h))
            else:
                area = m * (1-h) / (2*h* (1-m))
            heatmap_col[m] = area
        heatmap_dict[h] = heatmap_col
    heatmap = pd.DataFrame.from_dict(heatmap_dict)
    # plot to show the relationship of h and m in theorem2 w.r.t. area above the E_fr line
    sns.set_theme(rc={'figure.figsize': (figsize, figsize)})
    sns.set_style("whitegrid")
    with sns.plotting_context("talk", font_scale=1):
        axes = plt.gca()
        ticks = [1, 10, 20,30, 40,50, 60,70 ,80, 90,99]
        axes.set(xlim=(0, 1), ylim=(0, 1), xticks=[0, .5, 1], yticks=[0, .5, 1])
        ax = sns.heatmap(data = heatmap,
                         square = True,
                         )
        ax.set(xlabel="Human", ylabel="AI")
        ax.xaxis.tick_top()
        ax.yaxis.tick_left()
        plt.xticks(rotation=90)
        ax.set_yticks(ticks)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        # save the image
        dateTimeObj = datetime.now()
        time_stamp = dateTimeObj.strftime("%Y%m%d_%H%M")
        fig_name = '{}-{}.pdf'.format('heatmap_h_and_m', time_stamp)
        plt.savefig(Path('../result')/ fig_name, bbox_inches="tight")

    return heatmap









if __name__ == "__main__":
    # plot the images in the paper
    plot1 = PlotTrials(human_sim_acc=0.9,
                      ai_sim_acc=0.8,
                      plausibility_function=plau_corr_gaussian,
                      human_factor_function=fp_sigmoid,
                      )
    df1 = plot1.plot()
    plot2 = PlotTrials(human_sim_acc=0.8,
                      ai_sim_acc=0.9,
                      plausibility_function=plau_corr_gaussian,
                      human_factor_function=fp_sigmoid,
                      )
    df2 = plot2.plot()
    plot_h_m_relationship()



