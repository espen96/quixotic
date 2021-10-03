#version 150

#moj_import <fog.glsl>
#moj_import <utils.glsl>
uniform sampler2D Sampler0;

uniform mat4 ProjMat;
uniform vec4 ColorModulator;
uniform float FogStart;
uniform float FogEnd;
uniform vec4 FogColor;
in vec4 vertexColor;
in vec2 texCoord0;
in vec4 glpos;

out vec4 fragColor;

void main() {
    discardControlGLPos(gl_FragCoord.xy, glpos);

    vec4 color = texture(Sampler0, texCoord0) ;

  if (color.a*255 <= 17.0) {
    discard;
  }
    color.rgb = clamp(color.rgb,0.01,1);
    fragColor = vec4(ColorModulator.rgb * vertexColor.rgb, ColorModulator.a);
}
