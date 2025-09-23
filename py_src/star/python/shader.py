import OpenGL.GL as gl
import glfw
from PIL import Image
import numpy as np

# 1. Initialize GLFW and create a window (or off-screen context)
if not glfw.init():
    raise RuntimeError("GLFW initialization failed")
window = glfw.create_window(2048, 2048, "Shader Output", None, None)
if not window:
    glfw.terminate()
    raise RuntimeError("GLFW window creation failed")
glfw.make_context_current(window)

# 2. Define and compile shaders
vertex_shader_source = """
#version 330 core
layout (location = 0) in vec2 aPos;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"""

fragment_shader_source = """
#version 330 core
out vec4 FragColor;
void main() {
    FragColor = vec4(gl_FragCoord.xy / 512.0, 0.0, 1.0); // Example: color based on pixel coordinates
}
"""

vertex_shader = gl.glCreateShader(gl.GL_VERTEX_SHADER)
gl.glShaderSource(vertex_shader, vertex_shader_source)
gl.glCompileShader(vertex_shader)

fragment_shader = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
gl.glShaderSource(fragment_shader, fragment_shader_source)
gl.glCompileShader(fragment_shader)

shader_program = gl.glCreateProgram()
gl.glAttachShader(shader_program, vertex_shader)
gl.glAttachShader(shader_program, fragment_shader)
gl.glLinkProgram(shader_program)

gl.glDeleteShader(vertex_shader)
gl.glDeleteShader(fragment_shader)

# 3. Create FBO and texture
fbo = gl.glGenFramebuffers(1)
gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)

texture = gl.glGenTextures(1)
gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, 2048, 2048, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, texture, 0)

if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
    print("Framebuffer not complete!")

# 4. Render to FBO
gl.glUseProgram(shader_program)
gl.glViewport(0, 0, 2048, 2048)
gl.glClearColor(0.0, 0.0, 0.0, 1.0)
gl.glClear(gl.GL_COLOR_BUFFER_BIT)

# Define a full-screen quad
quad_vertices = np.array([
    -1.0, -1.0,
     1.0, -1.0,
     1.0,  1.0,
    -1.0, -1.0,
     1.0,  1.0,
    -1.0,  1.0
], dtype=np.float32)

vao = gl.glGenVertexArrays(1)
gl.glBindVertexArray(vao)
vbo = gl.glGenBuffers(1)
gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
gl.glBufferData(gl.GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, gl.GL_STATIC_DRAW)
gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 2 * quad_vertices.itemsize, None)
gl.glEnableVertexAttribArray(0)

gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)

# 5. Read pixel data
pixel_data = gl.glGetTexImage(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
image_array = np.frombuffer(pixel_data, dtype=np.uint8).reshape(2048, 2048, 4)

# 6. Save image
img = Image.fromarray(image_array)
img.save("./py_src/star/python/output/shader_output.png")

# Clean up
gl.glDeleteProgram(shader_program)
gl.glDeleteFramebuffers(1, [fbo])
gl.glDeleteTextures(1, [texture])
gl.glDeleteVertexArrays(1, [vao])
gl.glDeleteBuffers(1, [vbo])
glfw.terminate()