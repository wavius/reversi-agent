BUILD_DIR = build
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -I.
TARGET = $(BUILD_DIR)/reversi

SRCS = main.cpp
OBJS = $(patsubst %.cpp, $(BUILD_DIR)/%.o, $(SRCS))

# Pybind11 configuration
PYBIND_INCLUDES = $(shell python3 -m pybind11 --includes)
PYBIND_SUFFIX = $(shell python3 -m pybind11 --extension-suffix)
PY_TARGET = reversi_env$(PYBIND_SUFFIX)

.PHONY: all clean run python

all: $(TARGET)

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

python: bindings.cpp
	$(CXX) -O3 -Wall -shared -std=c++17 -fPIC $(PYBIND_INCLUDES) -I. bindings.cpp -o $(PY_TARGET)

clean:
	rm -rf $(BUILD_DIR) $(PY_TARGET)

run: $(TARGET)
	./$(TARGET)
