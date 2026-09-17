CXX = clang++

CXXFLAGS = -std=c++17 -O0 -Wall -Wextra \
           $(shell pkg-config --cflags opencv4)

LIBS = $(shell pkg-config --libs opencv4)

DEPS =

OBJ = readAndDisplayVideo.o

%.o: %.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

readAndDisplayVideo: $(OBJ)
	$(CXX) -o $@ $^ $(LIBS)

lab3: lab3.o
	$(CXX) -o $@ $^ $(LIBS)

lab4: lab4.o
	$(CXX) -o $@ $^ $(LIBS)

.PHONY: clean run run3 run4

run: readAndDisplayVideo
	./readAndDisplayVideo

run3: lab3
	./lab3

run4: lab4
	./lab4

clean:
	rm -f $(OBJ) lab3.o lab4.o readAndDisplayVideo lab3 lab4