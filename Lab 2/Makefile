CXX = clang++

CXXFLAGS = -std=c++17 -O2 -Wall -Wextra \
           $(shell pkg-config --cflags opencv4)

LIBS = $(shell pkg-config --libs opencv4)

DEPS =

OBJ = readAndDisplayVideo.o

%.o: %.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

readAndDisplayVideo: $(OBJ)
	$(CXX) -o $@ $^ $(LIBS)

.PHONY: clean run

run: readAndDisplayVideo
	./readAndDisplayVideo

clean:
	rm -f $(OBJ) readAndDisplayVideo