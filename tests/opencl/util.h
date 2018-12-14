#ifndef UTIL_H
#define UTIL_H

#include <stddef.h>
#include <CL/cl.h>

size_t load_file(const char *filename, char ***lines);

static inline const char *clDeviceTypeGetString(cl_device_type type) {
    switch (type) {
        case CL_DEVICE_TYPE_CPU: return "CL_DEVICE_TYPE_CPU";
        case CL_DEVICE_TYPE_GPU: return "CL_DEVICE_TYPE_CPU";
        case CL_DEVICE_TYPE_ACCELERATOR: return "CL_DEVICE_TYPE_ACCELERATOR";
        case CL_DEVICE_TYPE_DEFAULT: return "CL_DEVICE_TYPE_DEFAULT";
        case CL_DEVICE_TYPE_CUSTOM: return "CL_DEVICE_TYPE_CUSTOM";
        default: return "CL_DEVICE_UNKNOWN";
    }
}

#endif
