import matplotlib.pyplot as plt

from moment_subgrad.jikken1.agent import new_Agent, new_Agent_moment_CDC2017_paper, new_Agent_moment_CDC2017_paper2
from moment_subgrad.jikken1.iteration import new_iteration_L1


class new_iteration_L1_paper(new_iteration_L1):
    def make_agent(self, pattern):  # L1専用
        Agents = []
        s = self.step[pattern]
        for i in range(self.n):
            if pattern % 2 == 0:
                Agents.append(
                    new_Agent(self.n, self.m, self.A[i], self.p[i], s, self.lamb, name=i, weight=None, R=self.R))
            elif pattern % 2 == 1:
                Agents.append(
                    new_Agent_moment_CDC2017_paper(self.n, self.m, self.A[i], self.p[i], s, self.lamb, name=i,
                                                   weight=None, R=self.R))

        return Agents

class new_iteration_L1_paper_powerpoint(new_iteration_L1_paper):
    def make_graph(self, f_error):
        label = ['DSM', 'Proposed']
        line = ['-', '-.']
        for i in range(self.pattern):
            # stepsize = '_s(k)=' + str(self.step[i]) + '/k+10'
            stepsize = ' c=' + str(self.step[i])
            plt.plot(f_error[i], label=label[i % 2] + stepsize, linestyle=line[i % 2], linewidth=1)
        plt.legend()
        plt.xlim([1900,2000])
        plt.yscale('log')
        plt.xlabel('iteration $k$',fontsize=10)
        plt.ylabel('$max_{i}$ $f(x_i(k))-f^*$',fontsize=10)
        plt.show()


class new_iteration_L1_paper2(new_iteration_L1):
    def make_agent(self, pattern):  # L1専用
        Agents = []
        s = self.step[pattern]
        for i in range(self.n):
            if pattern % 2 == 0:
                Agents.append(
                    new_Agent(self.n, self.m, self.A[i], self.p[i], s, self.lamb, name=i, weight=None, R=self.R))
            elif pattern % 2 == 1:
                Agents.append(
                    new_Agent_moment_CDC2017_paper2(self.n, self.m, self.A[i], self.p[i], s, self.lamb, name=i,
                                                    weight=None, R=self.R))

        return Agents

    def make_graph(self, f_error):
        label = ['DSM', 'Proposed']
        line = ['-', '-.']
        for i in range(int(self.pattern)):
            # stepsize = '_s(k)=' + str(self.step[i]) + '/k+10'
            if i == 0:
                plt.plot(f_error[i], label=label[0], linestyle=line[0], linewidth=1)
            if i % 2 == 1:
                stepsize = ' gamma = ' + str(self.step[i])
                plt.plot(f_error[i], label=label[1] + stepsize, linestyle=line[1], linewidth=1)
        # plt.legend()
        # plt.yscale('log')
        # plt.xlabel('iteration $k$', fontsize=10)
        # plt.ylabel('$max_{i}  f(x_i(k))-f^*$',fontsize=10)
        # plt.show()
        plt.legend()
        plt.xlim([0, self.iterate])
        plt.yscale('log')
        plt.grid(which='major', color='black', linestyle='-')
        plt.grid(which='minor', color='gray', linestyle=':', axis='y')
        plt.minorticks_on()
        plt.xlabel('iteration $k$', fontsize=12)
        plt.ylabel('$max_{i}$ $f(x_i(k))-f^*$', fontsize=12)
        plt.tick_params(labelsize=12)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig("max_cost_comp_gamma.png")
        plt.savefig("max_cost_comp_gamma.eps")
        plt.show()
