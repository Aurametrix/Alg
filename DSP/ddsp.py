import ddsp

# Get synthesizer parameters from a neural network.
outputs = network(inputs)

# Initialize signal processors.
additive = ddsp.synths.Additive()

# Generates audio from additive synthesizer.
audio = additive(outputs['amplitudes'],
                 outputs['harmonic_distribution'],
                 outputs['f0_hz'])
