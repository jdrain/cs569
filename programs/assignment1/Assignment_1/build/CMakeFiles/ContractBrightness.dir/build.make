# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Produce verbose output by default.
VERBOSE = 1

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jason/school/cs569/programs/assignment1/Assignment_1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jason/school/cs569/programs/assignment1/Assignment_1/build

# Include any dependencies generated for this target.
include CMakeFiles/ContractBrightness.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ContractBrightness.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ContractBrightness.dir/flags.make

CMakeFiles/ContractBrightness.dir/ContractBrightness.cpp.o: CMakeFiles/ContractBrightness.dir/flags.make
CMakeFiles/ContractBrightness.dir/ContractBrightness.cpp.o: ../ContractBrightness.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jason/school/cs569/programs/assignment1/Assignment_1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ContractBrightness.dir/ContractBrightness.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ContractBrightness.dir/ContractBrightness.cpp.o -c /home/jason/school/cs569/programs/assignment1/Assignment_1/ContractBrightness.cpp

CMakeFiles/ContractBrightness.dir/ContractBrightness.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ContractBrightness.dir/ContractBrightness.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jason/school/cs569/programs/assignment1/Assignment_1/ContractBrightness.cpp > CMakeFiles/ContractBrightness.dir/ContractBrightness.cpp.i

CMakeFiles/ContractBrightness.dir/ContractBrightness.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ContractBrightness.dir/ContractBrightness.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jason/school/cs569/programs/assignment1/Assignment_1/ContractBrightness.cpp -o CMakeFiles/ContractBrightness.dir/ContractBrightness.cpp.s

CMakeFiles/ContractBrightness.dir/ContractBrightness.cpp.o.requires:

.PHONY : CMakeFiles/ContractBrightness.dir/ContractBrightness.cpp.o.requires

CMakeFiles/ContractBrightness.dir/ContractBrightness.cpp.o.provides: CMakeFiles/ContractBrightness.dir/ContractBrightness.cpp.o.requires
	$(MAKE) -f CMakeFiles/ContractBrightness.dir/build.make CMakeFiles/ContractBrightness.dir/ContractBrightness.cpp.o.provides.build
.PHONY : CMakeFiles/ContractBrightness.dir/ContractBrightness.cpp.o.provides

CMakeFiles/ContractBrightness.dir/ContractBrightness.cpp.o.provides.build: CMakeFiles/ContractBrightness.dir/ContractBrightness.cpp.o


# Object files for target ContractBrightness
ContractBrightness_OBJECTS = \
"CMakeFiles/ContractBrightness.dir/ContractBrightness.cpp.o"

# External object files for target ContractBrightness
ContractBrightness_EXTERNAL_OBJECTS =

ContractBrightness: CMakeFiles/ContractBrightness.dir/ContractBrightness.cpp.o
ContractBrightness: CMakeFiles/ContractBrightness.dir/build.make
ContractBrightness: /opt/opencv/build/lib/libopencv_objdetect.so.3.4.0
ContractBrightness: /opt/opencv/build/lib/libopencv_videostab.so.3.4.0
ContractBrightness: /opt/opencv/build/lib/libopencv_stitching.so.3.4.0
ContractBrightness: /opt/opencv/build/lib/libopencv_dnn.so.3.4.0
ContractBrightness: /opt/opencv/build/lib/libopencv_superres.so.3.4.0
ContractBrightness: /opt/opencv/build/lib/libopencv_calib3d.so.3.4.0
ContractBrightness: /opt/opencv/build/lib/libopencv_photo.so.3.4.0
ContractBrightness: /opt/opencv/build/lib/libopencv_shape.so.3.4.0
ContractBrightness: /opt/opencv/build/lib/libopencv_ml.so.3.4.0
ContractBrightness: /opt/opencv/build/lib/libopencv_video.so.3.4.0
ContractBrightness: /opt/opencv/build/lib/libopencv_features2d.so.3.4.0
ContractBrightness: /opt/opencv/build/lib/libopencv_highgui.so.3.4.0
ContractBrightness: /opt/opencv/build/lib/libopencv_flann.so.3.4.0
ContractBrightness: /opt/opencv/build/lib/libopencv_videoio.so.3.4.0
ContractBrightness: /opt/opencv/build/lib/libopencv_imgcodecs.so.3.4.0
ContractBrightness: /opt/opencv/build/lib/libopencv_imgproc.so.3.4.0
ContractBrightness: /opt/opencv/build/lib/libopencv_core.so.3.4.0
ContractBrightness: CMakeFiles/ContractBrightness.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jason/school/cs569/programs/assignment1/Assignment_1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ContractBrightness"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ContractBrightness.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ContractBrightness.dir/build: ContractBrightness

.PHONY : CMakeFiles/ContractBrightness.dir/build

CMakeFiles/ContractBrightness.dir/requires: CMakeFiles/ContractBrightness.dir/ContractBrightness.cpp.o.requires

.PHONY : CMakeFiles/ContractBrightness.dir/requires

CMakeFiles/ContractBrightness.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ContractBrightness.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ContractBrightness.dir/clean

CMakeFiles/ContractBrightness.dir/depend:
	cd /home/jason/school/cs569/programs/assignment1/Assignment_1/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jason/school/cs569/programs/assignment1/Assignment_1 /home/jason/school/cs569/programs/assignment1/Assignment_1 /home/jason/school/cs569/programs/assignment1/Assignment_1/build /home/jason/school/cs569/programs/assignment1/Assignment_1/build /home/jason/school/cs569/programs/assignment1/Assignment_1/build/CMakeFiles/ContractBrightness.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ContractBrightness.dir/depend

