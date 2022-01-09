#version 150

#moj_import <utils.glsl>
#moj_import <fog.glsl>
uniform sampler2D Sampler0;
uniform sampler2D Sampler2;
uniform float FogStart;
uniform float FogEnd;
uniform vec4 FogColor;
uniform vec4 ColorModulator;
in float vertexDistance;
in vec4 vertexColor;
in vec2 texCoord0;
in vec2 texCoord2;
in vec4 normal;
in vec4 glpos;

out vec4 fragColor;

void main() {
    discardControlGLPos(gl_FragCoord.xy, glpos);

    vec4 color = texture(Sampler0, texCoord0) * vertexColor * ColorModulator;
    if (color.a < 0.1) {
        discard;
    }
    fragColor = color;
    fragColor.a *=1- clamp(vertexDistance*0.003,0.0,1);
}
