#version 150

#moj_import <fog.glsl>
#moj_import <utils.glsl>

uniform sampler2D Sampler0;
uniform sampler2D Sampler1;

uniform vec4 ColorModulator;
uniform vec2 ScreenSize;
uniform float FogStart;
uniform float FogEnd;
uniform vec4 FogColor;

in float vertexDistance;
in vec4 vertexColor;
in vec2 texCoord0;
in vec2 texCoord1;
in vec4 normal;
in vec4 glpos;

out vec4 fragColor;

float luma(vec3 color){
	return dot(color,vec3(0.299, 0.587, 0.114));
}


void main() {
    	float aspectRatio = ScreenSize.x/ScreenSize.y;

    if (gl_PrimitiveID >= 2) {

            gl_FragDepth = 0.0;
    }
    fragColor = texture(Sampler0, vec2((gl_FragCoord.xy/ScreenSize)/ vec2(1,aspectRatio)) );    




//    fragColor.a *= (luma(FogColor.rgb)*luma(FogColor.rgb));
 //   fragColor.a *=  1-(fogValue*0.25);
}
