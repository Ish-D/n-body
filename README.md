# n-body
Realtime 3D N-Body Simulator that utilizes CUDA and Vulkan interoperability.
CUDA to implement the simulation logic using an all-pairs method. Vulkan to render the data that is exported from CUDA and synchronized using timeline semaphores.

# Examples
![](https://media.discordapp.net/attachments/812554947102113842/1147460770905468938/Peek_2023-09-02_01-51.gif)
![](https://cdn.discordapp.com/attachments/689615116122980483/1147631816375795732/Peek_2023-09-02_13-36.gif)
(Runs smoother than pictured with the weird stutter, recording gifs on Ubuntu is weird)

# Requirements
Requires GLFW, CUDA, and the Vulkan SDK to build.

# Building 
This project uses CMake to build, simply:

```
mkdir build
cd build
cmake ..
make
./nBody
```

Project has only been tested on Linux, no other operating systems thus far.
