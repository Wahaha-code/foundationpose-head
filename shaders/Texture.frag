#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D u_Texture;

void main()
{
    vec4 color = texture(u_Texture, vec2(TexCoord.x, -TexCoord.y));
    FragColor = color;
}