import torch 

class ClientState:
    def __init__(self, model, lmbd = 0.1, gamma = 0.1):
        self.weights = []
        self.state = [torch.zeros_like(p) for p in model.parameters()]
        self.prev_state = [torch.zeros_like(p) for p in model.parameters()]
        self.lmbd = lmbd
        self.gamma = gamma
    
    def local_step(self, model):
        for idx, param in enumerate(model.parameters()):
            self.state[idx] += param.grad - self.prev_state[idx]
            self.prev_state[idx] = param.grad
            # param.grad = torch.zeros_like(param.grad)
            param.grad.mult_(0.0)
    
    def global_step(self, model):
        for idx, param in enumerate(model.parameters()):
            self.state[idx] = param.grad
            self.prev_state[idx] = param.grad
            # param.grad = torch.zeros_like(param.grad)
            param.grad.data.zero_()

    def set_weights(self, model):
        self.weights = [p for p in model.parameters()]

    def get_reg_term(self, model):
        l2_regularization = torch.tensor(0., requires_grad=True)
        for idx, (param, x, g) in enumerate(zip(model.parameters(), self.weights, self.state)):
            l2_regularization = l2_regularization + ((param - (x - self.gamma * self.lmbd * g)) ** 2).sum()
        return l2_regularization
    
class AdamClientState:
    def __init__(self, model, lmbd = 0.1, gamma = 0.1):
        self.weights = []
        self.state = [torch.zeros_like(p) for p in model.parameters()]
        self.m_t = [torch.zeros_like(p) for p in model.parameters()]
        self.v_t = [torch.zeros_like(p) for p in model.parameters()]
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.step_idx = 1
        self.beta_1_t = self.beta_1 
        self.beta_2_t = self.beta_2
        self.epsilon = 1e-9
        self.lmbd = lmbd
        self.gamma = gamma
    
    def global_step(self, model):
        for idx, param in enumerate(model.parameters()):
            self.m_t[idx] = self.beta_1 * self.m_t[idx] + (1 - self.beta_1) * param.grad
            self.v_t[idx] = self.beta_2 * self.v_t[idx] + (1 - self.beta_2) * param.grad ** 2

            self.state[idx] =  (self.m_t[idx] / (1 - self.beta_1_t)) / (torch.sqrt(self.v_t[idx] / (1 - self.beta_2_t) + self.epsilon))
            self.beta_1_t *= self.beta_1
            self.beta_2_t *= self.beta_2
            # param.grad = torch.zeros_like(param.grad
            param.grad.data.zero_()
    
    def set_weights(self, model):
        self.weights = [p.detach() for p in model.parameters()]

    def get_reg_term(self, model):
        l2_regularization = torch.tensor(0., requires_grad=True)
        for idx, (param, x, g) in enumerate(zip(model.parameters(), self.weights, self.state)):
            l2_regularization = l2_regularization + ((param - (x.detach() - self.gamma * self.lmbd * g.detach())) ** 2).sum()
        return l2_regularization * 0.5
