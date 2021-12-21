#version 150

#moj_import <fog.glsl>
#moj_import <utils.glsl>

uniform sampler2D Sampler0;
uniform sampler2D Sampler1;

uniform vec4 ColorModulator;
uniform float FogStart;
uniform float FogEnd;
uniform vec4 FogColor;

in mat4 ProjMat2;
in float vertexDistance;
in vec4 vertexColor;
in vec4 lightMapColor;
in vec4 overlayColor;
in vec2 texCoord0;
in vec4 normal;
in float lm;
out vec4 fragColor;
in float lmx;
in float lmy;
in vec4 glpos;

void main() {
  vec3 rnd = ScreenSpaceDither(gl_FragCoord.xy);
  discardControlGLPos(gl_FragCoord.xy, glpos);
  vec4 color = texture(Sampler0, texCoord0) * vertexColor * ColorModulator;
bool gui = isGUI( ProjMat2);
  if(color.a * 255 <= 17.0) {
    discard;
  }
  color.rgb = mix(overlayColor.rgb, color.rgb, overlayColor.a);

//    color *= lightMapColor;

  color.rgb += rnd / 255;
  float mod2 = gl_FragCoord.x + gl_FragCoord.y;
  float res = mod(mod2, 2.0f);
  color.rgb = clamp(color.rgb, 0.01, 1);

  float lm = lmx;

if( vertexDistance < 1.5 && FogStart*0.000001 > 1) color.rgb *= lightMapColor.rgb;
  else if(res == 0.0f && !gui) {
    lm = lmy;
    color.b =  clamp(lmx, 0, 0.95);
    color.r =  clamp(lmy, 0, 0.95);
  }
//if( vertexDistance < 1.5) color.rgb = mix(overlayColor.rgb, color.rgb, overlayColor.a);
//  fragColor = linear_fog(color, vertexDistance, FogStart, FogEnd, FogColor);
  fragColor = color;
  //fragColor.rgb = vec3(normal);
  fragColor.a = packUnorm2x4(0.0, clamp(lm + (Bayer256(gl_FragCoord.xy) / 16), 0, 0.9));

}
