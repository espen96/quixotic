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
    vec3 rnd = ScreenSpaceDither( gl_FragCoord.xy );
    discardControlGLPos(gl_FragCoord.xy, glpos);
    vec4 color = texture(Sampler0, texCoord0) * vertexColor * ColorModulator;

  if (color.a*255 <= 17.0) {
    discard;
  }
  if(luma4(color.rgb) == 0 && color.a < 0.9) {
    color.rgb = vec3(color.a)* vertexColor.rgb * ColorModulator.rgb;
    if (color.a*255 <= 50.0) {
    discard;
  }
  }
    color.rgb = mix(overlayColor.rgb, color.rgb, overlayColor.a);
    color *= lightMapColor;

    color.rgb +=rnd/255;   
    color.rgb = clamp(color.rgb,0.01,1);


//  fragColor = linear_fog(color, vertexDistance, FogStart, FogEnd, FogColor);
    fragColor = vec4(color.rgb,1);

//    fragColor.a = packUnorm2x4( 0.0,clamp(lm+(Bayer256(gl_FragCoord.xy)/16),0,0.9));

}
