import numpy as np
import torch

def time_sine_uninformed(t):
    T_freq = [10]
    n_days = 1
    while (T_freq[-1] < 60*24*7*n_days):
        T_freq.append(T_freq[-1] * 2)
    omegas = [2*np.pi/t_freq for t_freq in T_freq]
    enc = []
    for om in omegas:
        enc = enc + [np.sin(om*t), np.cos(om*t)]
    return torch.Tensor(enc)

def time_sine_informed(t):
    # T_freq = [60, 60*12, 60*24]
    T_freq = [10, 30, 60, 60*3, 60*6, 60*12, 60*24]
    omegas = torch.Tensor([2*np.pi/t_freq for t_freq in T_freq])
    return torch.cat([torch.sin((om*t).unsqueeze(-1)) for om in omegas]+[torch.cos((om*t).unsqueeze(-1)) for om in omegas], dim=-1)


def time_sine_informed_few(t):
    T_freq = [60, 60*12, 60*24]
    omegas = [2*np.pi/t_freq for t_freq in T_freq]
    enc = []
    for om in omegas:
        enc = enc + [np.sin(om*t), np.cos(om*t)]
    return torch.Tensor(enc)


def time_sine_informed_many(t):
    T_freq = [10, 30, 60, 60*2, 60*4, 60*6, 60*8, 60*10, 60*12, 60*16, 60*20, 60*24]
    omegas = [2*np.pi/t_freq for t_freq in T_freq]
    enc = []
    for om in omegas:
        enc = enc + [np.sin(om*t), np.cos(om*t)]
    return torch.Tensor(enc)

def time_external(t):
    in_t = t
    mins = in_t % 60
    in_t = in_t // 60
    hrs = in_t % 24
    in_t = in_t // 24
    days = in_t % 7
    in_t = in_t // 7
    weeks = in_t
    return torch.Tensor([weeks, days, hrs, mins])

def time_linear(t):
    return torch.Tensor([t])

def time_semantic(t):
    clock_time = time_external(t)
    morning = (clock_time[2]<12).to(float)
    aft = (clock_time[2]>=12 and clock_time[2]<18).to(float)
    evening = (clock_time[2]>=18).to(float)
    return torch.Tensor([t, morning, aft, evening])

def time_external_normalized(t):
    in_t = t
    mins = in_t % 60
    in_t = in_t // 60
    hrs = in_t % 24
    in_t = in_t // 24
    days = in_t % 7
    in_t = in_t // 7
    weeks = in_t
    return torch.Tensor([weeks, days/7, hrs/24, mins/60])

def clock_time_with_weekend_bit(t, weekend_days):
    in_t = t
    mins = in_t % 60
    in_t = in_t // 60
    hrs = in_t % 24
    in_t = in_t // 24
    days = in_t % 7
    weekend = 1 if days in weekend_days else 0
    return torch.Tensor([weekend, hrs, mins])

class TimeEncodingOptions():
    def __init__(self, weekend_days=None):
        self.weekend_days = weekend_days
    def __call__(self, encoder_option):
        if encoder_option == 'sine_uninformed':
            return time_sine_uninformed
        elif encoder_option == 'sine_informed':
            return time_sine_informed
        elif encoder_option == 'sine_informed_few':
            return time_sine_informed_few
        elif encoder_option == 'sine_informed_many':
            return time_sine_informed_many
        elif encoder_option == 'linear':
            return time_linear
        elif encoder_option == 'semantic':
            return time_semantic
        elif encoder_option == 'external':
            return time_external
        elif encoder_option == 'external_normalized':
            return time_external_normalized
        elif encoder_option == 'clock_time_with_weekend_bit':
            assert self.weekend_days is not None, 'Require weekend days list to use clock_time_with_weekend_bit encoder'
            return lambda t : clock_time_with_weekend_bit(t, self.weekend_days)
        else:
            raise LookupError('Time encoding option is invalid!')


def human_readable_from_external(t):
    t=t.squeeze()
    weeks, days, hrs, mins = int(t[0]), int(t[1]), int(t[2]), int(t[3])
    t_h = '{:02d}:{:02d}'.format(hrs,mins)
    t_h = 'Week '+str(weeks)+', Day '+str(days)+', '+t_h
    return t_h