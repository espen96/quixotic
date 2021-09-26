#version 150

#moj_import <fog.glsl>
#moj_import <utils.glsl>

uniform sampler2D Sampler0;

uniform vec4 ColorModulator;
uniform float FogStart;
uniform float FogEnd;
uniform vec4 FogColor;

in float vertexDistance;
in float water;
in vec4 vertexColor;
in vec3 noise;
in vec3 color2;
in vec2 texCoord0;
in vec4 normal;
in vec4 glpos;

out vec4 fragColor;






void main() {
    discardControlGLPos(gl_FragCoord.xy, glpos);
    
    vec4 color = texture(Sampler0, texCoord0) * vertexColor * ColorModulator;
    

    fragColor = linear_fog(color, vertexDistance,FogStart, FogEnd, FogColor);


 // color.rgb = (noise);
    if(water > 0.9 )    fragColor = linear_fog(color, vertexDistance, -8, FogEnd*0.1, FogColor);

//    if(water > 0.9 )  fragColor.a = 0.75;

    
}
