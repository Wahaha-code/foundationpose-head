#version 330 core

precision mediump float;

uniform vec4 u_ColorCorrection;
uniform vec4 u_MaterialParameters;
uniform vec4 u_ObjColor;


varying vec3 v_ViewPosition;
varying vec3 v_ViewNormal;
varying vec3 v_ViewLightOri;


void main() 
{

    // We support approximate sRGB gamma.
    const float kGamma = 0.4545454;
    const float kInverseGamma = 2.2;
    const float kMiddleGrayGamma = 0.466;

    // Unpack lighting and material parameters for better naming.
    vec3 viewLightDirection = v_ViewLightOri;
    vec3 colorShift = u_ColorCorrection.rgb;
    float averagePixelIntensity = u_ColorCorrection.a;

    float materialAmbient = u_MaterialParameters.x;
    float materialDiffuse = u_MaterialParameters.y;
    float materialSpecular = u_MaterialParameters.z;
    float materialSpecularPower = u_MaterialParameters.w;

    // Normalize varying parameters, because they are linearly interpolated in the vertex shader.
    vec3 viewFragmentDirection = normalize(-v_ViewPosition);
    vec3 viewNormal = normalize(v_ViewNormal);

    vec4 objectColor = u_ObjColor;

    // Apply inverse SRGB gamma to the texture before making lighting calculations.
    //do not use cottection now
    objectColor.rgb = pow(objectColor.rgb, vec3(kInverseGamma));

    // Ambient light is unaffected by the light intensity.
    float ambient = materialAmbient;

    float diffuse = materialDiffuse * max(dot(viewNormal, -viewLightDirection), 0.0);

    // Compute specular light.
    vec3 reflectedLightDirection = reflect(viewLightDirection, viewNormal);
    float specularStrength = max(0.0, dot(viewFragmentDirection, reflectedLightDirection));
    float specular = materialSpecular * pow(specularStrength, materialSpecularPower);

    vec3 color = u_ObjColor.rgb * (ambient + diffuse) + specular;

    gl_FragColor.rgb = color;
    gl_FragColor.a = objectColor.a;

}
