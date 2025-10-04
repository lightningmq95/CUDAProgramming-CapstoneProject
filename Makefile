# Makefile for Windows with NVCC

# Compiler
NVCC = nvcc

# Target executable name
TARGET = image_processor.exe

# Source file
SRCS = main.cu

# Compiler flags
# -arch=sm_xx should be set to your GPU's compute capability.
# sm_75 is for Turing, sm_86 for Ampere. Check your GPU's capability.
CXXFLAGS = -O3 -arch=sm_75 -std=c++17

# Libraries to link
# NPP (Core, Image Filtering, Image Geometry) and cuFFT (Fourier Transform)
# <-- FIX: Added the NPP Image Filtering library (-lnppif)
LIBS = -lnppc -lnppif -lnppig -lcufft

# Default rule
all: $(TARGET)

# Rule to build the target executable
$(TARGET): $(SRCS)
	$(NVCC) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(LIBS)
	@echo "Compilation finished successfully. Run with: ./$(TARGET)"

# Clean rule to remove generated files
clean:
	del $(TARGET) *.o *.png *.lib *.exp