#version 330 core

uniform mat4 u_ModelView;
uniform mat4 u_ModelViewProjection;
uniform mat4 u_ViewMatrix;
uniform mat4 u_ModelMatrix;
uniform mat4 u_PerspectiveProjMatrix;
uniform vec3 u_LightOri;


attribute vec4 a_Position;
attribute vec3 a_Normal;
//attribute vec2 a_TexCoord;

varying vec3 v_ViewPosition;
varying vec3 v_ViewNormal;
varying vec3 v_ViewLightOri;

//the plane is defined in the RAS space
uniform vec4 u_clipPlane;

void main() {
    v_ViewPosition = (u_ViewMatrix * u_ModelMatrix * a_Position).xyz;
    v_ViewNormal = normalize((u_ViewMatrix * u_ModelMatrix * vec4(a_Normal, 0.0)).xyz);
    v_ViewLightOri = normalize((u_ViewMatrix * vec4(u_LightOri, 1.0f)).xyz);
    gl_Position = u_PerspectiveProjMatrix * u_ViewMatrix * u_ModelMatrix * a_Position;
}
