#version 150

#moj_import <fog.glsl>
#moj_import <utils.glsl>

uniform sampler2D Sampler0;

uniform vec4 ColorModulator;
uniform float FogStart;
uniform float FogEnd;

in float vertexDistance;
in vec2 texCoord0;

out vec4 fragColor;
in float lmx;
in float lmy;
in vec4 glpos;
void main() {
  discardControlGLPos(gl_FragCoord.xy, glpos);
  vec4 color = texture(Sampler0, texCoord0) * ColorModulator;

  if(color.a * 255 <= 17.0) {
    discard;
  }
    float mod2 = gl_FragCoord.x + gl_FragCoord.y;
  float res = mod(mod2, 2.0f);
if( FogStart*0.000001 > 1) color.rgb = color.rgb;
  else if(res == 0.0f ) {
    color.b =  clamp(lmx, 0, 0.95);
    color.r =  clamp(lmy, 0, 0.95);
  }


  color.rgb = clamp(color.rgb, 0.01, 1);
  float fade = linear_fog_fade(vertexDistance, FogStart, FogEnd);
  fragColor = vec4(color.rgb * fade, color.a);
}
