#version 150

uniform sampler2D DiffuseSampler;
uniform sampler2D DiffuseDepthSampler;
uniform sampler2D MainSampler;


uniform vec2 ScreenSize;
in float near;
in float far;
in float center;
in vec2 texCoord;
in vec2 oneTexel;

out vec4 fragColor;


	const vec2 hex_offsets[60] = vec2[60] (	vec2(  0.2165,  0.1250 ),
											vec2(  0.0000,  0.2500 ),
											vec2( -0.2165,  0.1250 ),
											vec2( -0.2165, -0.1250 ),
											vec2( -0.0000, -0.2500 ),
											vec2(  0.2165, -0.1250 ),
											vec2(  0.4330,  0.2500 ),
											vec2(  0.0000,  0.5000 ),
											vec2( -0.4330,  0.2500 ),
											vec2( -0.4330, -0.2500 ),
											vec2( -0.0000, -0.5000 ),
											vec2(  0.4330, -0.2500 ),
											vec2(  0.6495,  0.3750 ),
											vec2(  0.0000,  0.7500 ),
											vec2( -0.6495,  0.3750 ),
											vec2( -0.6495, -0.3750 ),
											vec2( -0.0000, -0.7500 ),
											vec2(  0.6495, -0.3750 ),
											vec2(  0.8660,  0.5000 ),
											vec2(  0.0000,  1.0000 ),
											vec2( -0.8660,  0.5000 ),
											vec2( -0.8660, -0.5000 ),
											vec2( -0.0000, -1.0000 ),
											vec2(  0.8660, -0.5000 ),
											vec2(  0.2163,  0.3754 ),
											vec2( -0.2170,  0.3750 ),
											vec2( -0.4333, -0.0004 ),
											vec2( -0.2163, -0.3754 ),
											vec2(  0.2170, -0.3750 ),
											vec2(  0.4333,  0.0004 ),
											vec2(  0.4328,  0.5004 ),
											vec2( -0.2170,  0.6250 ),
											vec2( -0.6498,  0.1246 ),
											vec2( -0.4328, -0.5004 ),
											vec2(  0.2170, -0.6250 ),
											vec2(  0.6498, -0.1246 ),
											vec2(  0.6493,  0.6254 ),
											vec2( -0.2170,  0.8750 ),
											vec2( -0.8663,  0.2496 ),
											vec2( -0.6493, -0.6254 ),
											vec2(  0.2170, -0.8750 ),
											vec2(  0.8663, -0.2496 ),
											vec2(  0.2160,  0.6259 ),
											vec2( -0.4340,  0.5000 ),
											vec2( -0.6500, -0.1259 ),
											vec2( -0.2160, -0.6259 ),
											vec2(  0.4340, -0.5000 ),
											vec2(  0.6500,  0.1259 ),
											vec2(  0.4325,  0.7509 ),
											vec2( -0.4340,  0.7500 ),
											vec2( -0.8665, -0.0009 ),
											vec2( -0.4325, -0.7509 ),
											vec2(  0.4340, -0.7500 ),
											vec2(  0.8665,  0.0009 ),
											vec2(  0.2158,  0.8763 ),
											vec2( -0.6510,  0.6250 ),
											vec2( -0.8668, -0.2513 ),
											vec2( -0.2158, -0.8763 ),
											vec2(  0.6510, -0.6250 ),
											vec2(  0.8668,  0.2513 ));


//lens properties
#define focal  3.5
#define aperture 2.8
#define MANUAL_FOCUS 1.0
//#define dof

float ld(float depth) {
    return (2.0 * near) / (far + near - depth * (far - near));		// (-depth * (far - near)) = (2.0 * near)/ld - far - near
}



const float PI_3 = 1.0471975512;
#define hexa(k) vec2(cos(PI_3 * k), sin(PI_3 * k))
const vec2 paint_offsets[6] = vec2[6] (
    hexa(0.), hexa(1.), hexa(2.),
    hexa(3.), hexa(4.), hexa(5.));
float luma(vec3 color){
	return dot(color,vec3(0.299, 0.587, 0.114));
}

void main() {
    vec3 currentColor = texture(DiffuseSampler, texCoord).rgb;
    float midDepth = ld(texture(DiffuseDepthSampler, vec2(0.5, 0.5)).r)*far;
          midDepth = MANUAL_FOCUS;

    float aspectRatio = ScreenSize.x/ScreenSize.y;
    
	  float depth = ld(texture( DiffuseDepthSampler, texCoord ).r)*far; 
    vec3 color = texture( DiffuseSampler, texCoord ).rgb;

#ifdef dof
		vec3 bcolor = vec3(0.);
		vec3 bcolor2 = vec3(0.);
		float nb = 0.0;
		vec2 bcoord = vec2(0.0);
		/*--------------------------------*/
/*
		bcolor = color;
		float pcoc = min(abs(aperture * (focal/100.0 * (depth - midDepth)) / (depth * (midDepth - focal/100.0))),oneTexel.x*15.0);
		//	  pcoc *= float(depth > midDepth);

		vec3 t = vec3 (0.0);
		float h = 1.004;
		vec2 d = vec2(pcoc / aspectRatio, pcoc);

		for ( int i = 0; i < 60; i++) {

				bcolor += texture(DiffuseSampler, texCoord.xy + hex_offsets[i]*pcoc*vec2(1.0,aspectRatio)).rgb;
				
			}
			color = bcolor/61.0;


	*/

	
		
				bcolor = color;
				bcolor2 = color;

			float pcoc = min(abs(aperture * (focal/100.0 * (depth - midDepth)) / (depth * (midDepth - focal/100.0))),oneTexel.x*15.0);
			      pcoc *= float(depth > midDepth);
		//		  pcoc =  pow(pcoc,2)*100;
				vec3 t = vec3 (0.0);
   				float v = 0.004; // spot size
				  pcoc = clamp(pcoc,0.0,v);

				vec2 d = vec2(v / aspectRatio, v);
				vec2 xy = vec2(texCoord.x, 1.0f - texCoord.y);
		
				for (int i = 0; i < 6; ++i) {
					bcolor = texture(DiffuseSampler, texCoord.xy + paint_offsets[i]*d).rgb;
					 t = max(sign(bcolor - bcolor2), 0.0);
				
				bcolor2 += (bcolor - bcolor2) * t;
				}
				 color = bcolor2;

			
#endif
				
	  fragColor = vec4(vec3(color), 1.0 );    
}
