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
in vec4 lm;
in vec4 glpos;
in float lmx;
in float lmy;
out vec4 fragColor;

void main() {

    float mod2 = gl_FragCoord.x + gl_FragCoord.y;
    float res = mod(mod2, 2.0f);
    discardControlGLPos(gl_FragCoord.xy, glpos);
    vec4 color = texture(Sampler0, texCoord0);
    color = color * vertexColor * ColorModulator;

    if(textureLod(Sampler0, texCoord0,0).a *255 == 200) {
    color.a = textureLod(Sampler0, texCoord0,0).a;
    color.rgb = mix(color.rgb,vec3(0.0),0.5);
    }



  
if( FogStart*0.000001 > 1) color.rgb = color.rgb;
  else if(res == 0.0f ) {
    color.b =  clamp(lmx, 0, 0.95);
    color.r =  clamp(lmy, 0, 0.95);
  }


    color.rgb = clamp(color.rgb, 0.01, 1);
    fragColor = color;


}
