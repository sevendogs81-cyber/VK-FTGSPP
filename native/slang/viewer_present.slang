struct DisplayParams {
    uint width; uint height; uint mode; uint checker;
    float2 origin; float2 extent;
    float4 background;
    float exposure; uint linear_output; float2 pad;
};
[[vk::push_constant]] ConstantBuffer<DisplayParams> p;
[[vk::binding(0,0)]] StructuredBuffer<float4> pixel_ft;
[shader("vertex")]
float4 viewer_vs(uint id : SV_VertexID) : SV_Position {
    float2 uv=float2((id<<1)&2,id&2);
    return float4(uv*2-1,0,1);
}
float4 fetch(int2 xy) {
    xy=clamp(xy,int2(0),int2(p.width,p.height)-1);
    return pixel_ft[xy.y*p.width+xy.x];
}
[shader("fragment")]
float4 viewer_fs(float4 pos : SV_Position) : SV_Target0 {
    float2 xy=(pos.xy-p.origin)/p.extent*float2(p.width,p.height)-0.5;
    float4 ft;
    if (all(abs(p.extent-float2(p.width,p.height))<0.01)) ft=fetch(int2(floor(xy+0.5)));
    else {
        int2 ij=int2(floor(xy));float2 f=frac(xy);
        ft=lerp(lerp(fetch(ij),fetch(ij+int2(1,0)),f.x),lerp(fetch(ij+int2(0,1)),fetch(ij+1),f.x),f.y);
    }
    float t=saturate(ft.w);
    float3 bg=p.background.xyz;
if(p.checker!=0) bg=float3((((uint(pos.x/16)+uint(pos.y/16))&1)!=0)?0.65:0.35);
    float3 c=p.mode==1?float3(1-t):p.mode==2?float3(t):ft.xyz+t*bg;
    if(p.mode==0)c*=exp2(p.exposure);
    c=saturate(c);
    if(p.linear_output!=0)c=select(c<=0.04045,c/12.92,pow((c+0.055)/1.055,2.4));
    return float4(c,1);
}
