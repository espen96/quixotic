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
  //vec3 rnd = ScreenSpaceDither(gl_FragCoord.xy);
  //color.rgb += rnd / 255;
  if(color.a * 255 <= 17.0) {
    discard;
  }
    float mod2 = gl_FragCoord.x + gl_FragCoord.y;
  float res = mod(mod2, 2.0f);

  /*
if( FogStart*0.000001 > 1) color.rgb = color.rgb;
  else if(res == 0.0f ) {
    color.b =  clamp(lmx, 0, 0.95);
    color.r =  clamp(lmy, 0, 0.95);
  }
*/
  color.rgb = clamp(color.rgb, 0.01, 1);

//  fragColor = linear_fog(color, vertexDistance, FogStart, FogEnd, FogColor);
  fragColor = vec4(color.rgb, (color.a));
}
