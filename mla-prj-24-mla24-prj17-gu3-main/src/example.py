
from architecture.original.network_system import LMU, L2MU
import torch


if __name__ == "__main__":

    input_tensor = torch.randn(40, 256, 6) # (sequence_length, batch_size, feature_input)

    params_lmu = {
        "hidden_size": 20,
        "memory_size": 20,
        "order": 3,
        "theta": 1,
    }

    num_classes_to_predict = 10
    lmu_model = LMU(input_size=6, output_size=num_classes_to_predict, params=params_lmu)
    lmu_model.eval()
    output = lmu_model(input_tensor)

    print(f'LMU output: {output}')

    params_l2mu_leaky = {
        "hidden_size": 20,
        "memory_size": 20,
        "order": 3,
        "theta": 1,
        "beta_spk_u": 0.4,
        "threshold_spk_u": 0.15,
        "beta_spk_h": 0.2,
        "threshold_spk_h": 0.65,
        "beta_spk_m": 0.55,
        "threshold_spk_m": 0.9,
        "beta_spk_output": 0.7,
        "threshold_spk_output": 0.75
    }

    l2mu_leaky_model = L2MU(input_size=6, output_size=num_classes_to_predict, params=params_l2mu_leaky)
    l2mu_leaky_model.eval()
    output = l2mu_leaky_model(input_tensor)

    print(f'L2MU Leaky output: {output}')

