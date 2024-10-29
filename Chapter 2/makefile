# Define the compiler
CXX = g++
CXXFLAGS = -std=c++17 -I./include -Wall -Wextra -Werror

# Data
DATA_FOLDER = data

# Executables
MAIN_EXECUTABLE = build\main.exe
EXAMPLE_EXECUTABLE = build\generate_k_normal_distribution_samples.exe

# Default targets
all: $(MAIN_EXECUTABLE) $(EXAMPLE_EXECUTABLE)

# Rule to compile the main program
$(MAIN_EXECUTABLE): src/*.cc
	$(CXX) $(CXXFLAGS) -o $@ $^

# Rule to compile the example program
$(EXAMPLE_EXECUTABLE): examples/*.cc
	$(CXX) $(CXXFLAGS) -o $@ $^

# Rule to clean up generated files
clean:
	del $(MAIN_EXECUTABLE) $(EXAMPLE_EXECUTABLE)
	del /F /Q /S $(DATA_FOLDER)\*

.PHONY: all clean