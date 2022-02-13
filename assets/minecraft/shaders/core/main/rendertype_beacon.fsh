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
in float lmx;
in float lmy;
out vec4 fragColor;

void main() {
  discardControlGLPos(gl_FragCoord.xy, glpos);
  //vec3 rnd = ScreenSpaceDither(gl_FragCoord.xy);

  vec4 color = texture(Sampler0, texCoord0) * vertexColor;

  if(color.a * 255 <= 250.0) {
    discard;
  }
  //color.rgb += rnd / 255;
  color.rgb = clamp(color.rgb, 0.01, 1);
 // float fragmentDistance = -ProjMat[3].z / ((gl_FragCoord.z) * -2.0 + 1.0 - ProjMat[2].z);
//  fragColor = linear_fog(color, vertexDistance, FogStart, FogEnd, FogColor);
  float mod2 = gl_FragCoord.x + gl_FragCoord.y;
  float res = mod(mod2, 2.0f);
if(FogStart*0.000001 > 1) color.rgb = color.rgb;
  else if(res == 0.0f) {

    color.b =  0.95;
    color.r =  0.95;
  }
  

  fragColor = vec4(color.rgb, 0.22);
}
