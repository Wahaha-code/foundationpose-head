#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

uniform float u_AspectRatio;
uniform ivec2 u_ScreenSize;
out vec2 TexCoord;

void main()
{
    int screen_width = u_ScreenSize[0];
    int screen_height = u_ScreenSize[1];
    float aspect_ratio = 1.0f * screen_width / screen_height;
    vec3 adjustPos;
    //if (aspect_ratio >= u_AspectRatio)
    //{
        
    //    adjustPos = vec3(u_AspectRatio * abs(aPos.y) * sign(aPos.x) / aspect_ratio, aPos.yz);
    //}
    //else
    //{
    //    adjustPos = vec3(aPos.x, aspect_ratio * abs(aPos.x) * sign(aPos.y) / u_AspectRatio, aPos.z);
    //}
    adjustPos = vec3(u_AspectRatio * abs(aPos.y) * sign(aPos.x) / aspect_ratio, aPos.yz);
    gl_Position = vec4(adjustPos, 1.0);
    //gl_Position = vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}