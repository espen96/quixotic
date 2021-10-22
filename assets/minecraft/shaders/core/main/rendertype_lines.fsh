#version 150

#moj_import <fog.glsl>
#moj_import <utils.glsl>

uniform vec4 ColorModulator;
uniform float FogStart;
uniform float FogEnd;
uniform vec4 FogColor;

in float vertexDistance;
in vec4 vertexColor;

out vec4 fragColor;

in vec4 glpos;
void main() {
    discardControlGLPos(gl_FragCoord.xy, glpos);
    vec4 color = vertexColor * ColorModulator;
    color.rgb = clamp(color.rgb, 0.001, 1);
//  fragColor = linear_fog(color, vertexDistance, FogStart, FogEnd, FogColor);
    fragColor = color;

}
