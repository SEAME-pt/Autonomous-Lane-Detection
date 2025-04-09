CC = g++

TENSORRT_INC_DIR = /usr/include/aarch64-linux-gnu
TENSORRT_LIB_DIR = /usr/lib/aarch64-linux-gnu

CFLAGS = -Wall -Wextra -Werror -std=c++17 -O3 -g \
         -Iincludes \
         $(shell pkg-config --cflags opencv4) \
         -I$(TENSORRT_INC_DIR) \
         -I/usr/local/cuda/include

LDFLAGS = -lSDL2 -lSDL2_image -lSDL2_ttf -lSDL2_mixer -li2c -lpthread \
          $(shell pkg-config --libs opencv4) \
          -L/usr/lib/aarch64-linux-gnu -lnvinfer -lnvinfer_plugin \
          -L/usr/local/cuda/lib64 -lcudart -Wl,-rpath,/usr/local/cuda/lib64 \
          -lmosquitto

JETRACER_SERVER_SRC = sources/jetracer_server.cpp sources/jetracer.cpp
JETRACER_SERVER_OBJ = $(JETRACER_SERVER_SRC:.cpp=.o)
JETRACER_SERVER_EXEC = jetracer_server

LANE_DETECTION_SRC = sources/lane_detection.cpp
LANE_DETECTION_OBJ = $(LANE_DETECTION_SRC:.cpp=.o)
LANE_DETECTION_EXEC = lane_detection

all: $(JETRACER_SERVER_EXEC) $(LANE_DETECTION_EXEC)

$(JETRACER_SERVER_EXEC): $(JETRACER_SERVER_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(LANE_DETECTION_EXEC): $(LANE_DETECTION_OBJ) sources/jetracer.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(JETRACER_SERVER_OBJ) $(LANE_DETECTION_OBJ) $(JETRACER_SERVER_EXEC) $(LANE_DETECTION_EXEC)

run-server: $(JETRACER_SERVER_EXEC)
	./$(JETRACER_SERVER_EXEC)

run-lane: $(LANE_DETECTION_EXEC)
	./$(LANE_DETECTION_EXEC)

kill-process:
	sudo lsof -ti :65432 | xargs -r sudo kill -9

re: clean all

.PHONY: all clean run-server run-lane re

