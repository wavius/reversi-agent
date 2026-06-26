CXX = emcc
CXXFLAGS = -std=c++17 -O3 -flto -I../game
LDFLAGS = --bind -s WASM=1 -s ALLOW_MEMORY_GROWTH=1 -s MODULARIZE=1 -s EXPORT_NAME="createReversiModule"

TARGET = reversi_wasm.js

all: $(TARGET)

$(TARGET): bindings_web.cpp
	$(CXX) $(CXXFLAGS) bindings_web.cpp -o $(TARGET) $(LDFLAGS)

clean:
	rm -f reversi_wasm.js reversi_wasm.wasm
