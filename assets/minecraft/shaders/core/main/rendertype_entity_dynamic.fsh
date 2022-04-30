#version 150

#moj_import <fog.glsl>
#moj_import <utils.glsl>

uniform sampler2D Sampler0;
uniform float FogStart;
uniform vec4 ColorModulator;
in float lmx;
in float lmy;
in vec2 texCoord0;
in vec4 vertexColor;
out vec4 fragColor;
in vec4 glpos;
void main() {
  discardControlGLPos(gl_FragCoord.xy, glpos);
  vec4 color = texture(Sampler0, texCoord0) * vertexColor * ColorModulator;

  if(color.a * 255 <= 17.0) {
    discard;
  }


  color.rgb = clamp(color.rgb, 0.01, 1);
  fragColor = vec4(color.rgb, (color.a));
}
