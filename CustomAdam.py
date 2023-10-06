from torch.optim import Optimizer


class CustomAdam(Optimizer):
    def __init__(self, params, t, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        # Initialize the hyperparameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = t + 1

        # Call the constructor of the parent class (Optimizer)
        super(CustomAdam, self).__init__(params, defaults={})

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                assert p.data.size(-1) % 3 == 0, "The parameter's size must be divisible by 3 (data, m, v)"
                size = p.data.size(-1) // 3

                # Retrieve gradients and the parameter itself
                grad = p.grad.data[..., :size]
                state = self.state[p]

                state['m'] = p.data[..., size:(size * 2)]
                state['v'] = p.data[..., (size * 2):]

                t = self.t

                # Exponential moving average of the gradients (momentum)
                state['m'] = p.data[..., size:(size * 2)] = self.beta1 * state['m'] + (1 - self.beta1) * grad

                # Exponential moving average of the squared gradients (uncentered variance)
                state['v'] = p.data[..., (size * 2):] = self.beta2 * state['v'] + (1 - self.beta2) * (grad ** 2)

                # Bias-corrected moving averages
                m_hat = state['m'] / (1 - self.beta1 ** t)
                v_hat = state['v'] / (1 - self.beta2 ** t)

                # Update the parameter using the computed moving averages
                p.data[..., :size] -= group['lr'] * m_hat / (v_hat.sqrt() + self.epsilon)
