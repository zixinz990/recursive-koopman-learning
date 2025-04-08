#pragma once

#define NUM_STATE_OBS 28 // Poly
// #define NUM_STATE_OBS 32 // Fourier
// #define NUM_STATE_OBS 121 // Gaussian RBF
#define NUM_INPUT_OBS 6

#define FEEDBACK_PERIOD 20.0 // ms
#define SERVO_CMD_PERIOD 20.0 // ms
#define MODEL_UPDATE_PERIOD 20.0 // ms
#define CONTROL_PERIOD 20.0 // ms
