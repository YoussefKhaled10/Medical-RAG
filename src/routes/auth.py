import re
import secrets
from typing import Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession
from src.dependencies.auth import get_current_user
from src.helpers.email_service import generate_otp_code, send_verification_email
from src.helpers.security import create_access_token, create_refresh_token, hash_password, verify_password
from src.models import AssetModel, ProjectModel, UserModel, get_db_session
from src.models.db_schemes.medical_rag import User

auth_router=APIRouter(prefix="/api/v1/auth",tags=["Authentication"])
EMAIL_RE=re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
class SendOTPRequest(BaseModel):
 model_config=ConfigDict(extra="forbid")
 email:str=Field(min_length=5,max_length=255)
 @field_validator("email")
 @classmethod
 def email_ok(cls,v):
  v=v.strip().lower()
  if not EMAIL_RE.fullmatch(v): raise ValueError("Invalid email")
  return v
class RegisterRequest(SendOTPRequest):
 username:str=Field(min_length=3,max_length=50,pattern=r"^[A-Za-z0-9_.-]+$")
 password:str=Field(min_length=8,max_length=128)
 full_name:str|None=Field(None,max_length=100)
 otp_code:str=Field(pattern=r"^\d{6}$")
class LoginRequest(BaseModel):
 email_or_username:str=Field(min_length=3,max_length=255); password:str
class RefreshTokenRequest(BaseModel): refresh_token:str
class LogoutRequest(BaseModel): refresh_token:str|None=None
class UserProfileResponse(BaseModel):
 id:int; email:str; username:str; full_name:str|None; is_verified:bool; is_admin:bool=False; private_project_id:int|None; vault_name:str|None=None; document_count:int=0
class AuthTokenResponse(BaseModel):
 access_token:str; refresh_token:str; token_type:str="bearer"; user:UserProfileResponse
async def profile(session,user):
 assets=await AssetModel.list_by_project(session,user.private_project_id) if user.private_project_id else []
 return UserProfileResponse(id=user.id,email=user.email,username=user.username,full_name=user.full_name,is_verified=user.is_verified,is_admin=bool(getattr(user,"is_admin",False)),private_project_id=user.private_project_id,vault_name=user.private_project.name if user.private_project else None,document_count=len(assets))
async def issue(session,user):
 access=create_access_token({"sub":str(user.id),"type":"access"}); refresh,expires=create_refresh_token(user.id)
 await UserModel.store_refresh_token(session,user.id,refresh,expires,commit=False); await session.commit()
 return AuthTokenResponse(access_token=access,refresh_token=refresh,user=await profile(session,user))
@auth_router.post("/send-otp")
async def send_otp(p:SendOTPRequest,session:AsyncSession=Depends(get_db_session))->dict[str,Any]:
 if await UserModel.get_by_email(session,p.email): raise HTTPException(409,"Email already registered")
 code=generate_otp_code(); await UserModel.create_verification_code(session,p.email,code,15,commit=False)
 if not await send_verification_email(p.email,code): await session.rollback(); raise HTTPException(502,"OTP email could not be sent")
 await session.commit(); return {"success":True,"expires_in_minutes":15}
@auth_router.post("/register",response_model=AuthTokenResponse,status_code=201)
async def register(p:RegisterRequest,session:AsyncSession=Depends(get_db_session)):
 if await UserModel.get_by_email(session,p.email): raise HTTPException(409,"Email already registered")
 if await UserModel.get_by_username(session,p.username): raise HTTPException(409,"Username already taken")
 if not await UserModel.verify_code(session,p.email,p.otp_code,True,False): raise HTTPException(400,"Invalid or expired OTP")
 try:
  project=await ProjectModel.create(session,name=f"vault_{p.username}_{secrets.token_hex(4)}",description=f"Private vault for {p.username}",commit=False)
  user=await UserModel.create_user(session,p.email,p.username,hash_password(p.password),p.full_name,project.id,True,commit=False)
  return await issue(session,user)
 except Exception:
  await session.rollback(); raise
@auth_router.post("/login",response_model=AuthTokenResponse)
async def login(p:LoginRequest,session:AsyncSession=Depends(get_db_session)):
 user=await UserModel.get_by_email_or_username(session,p.email_or_username)
 if not user or not verify_password(p.password,user.hashed_password): raise HTTPException(401,"Invalid credentials")
 if not user.is_active or not user.is_verified: raise HTTPException(403,"Account is unavailable or unverified")
 return await issue(session,user)
@auth_router.post("/refresh")
async def refresh(p:RefreshTokenRequest,session:AsyncSession=Depends(get_db_session)):
 rt=await UserModel.get_valid_refresh_token(session,p.refresh_token)
 if not rt: raise HTTPException(401,"Invalid or expired refresh token")
 return {"access_token":create_access_token({"sub":str(rt.user_id),"type":"access"}),"token_type":"bearer"}
@auth_router.get("/me",response_model=UserProfileResponse)
async def me(user:User=Depends(get_current_user),session:AsyncSession=Depends(get_db_session)): return await profile(session,user)
@auth_router.post("/logout")
async def logout(p:LogoutRequest,user:User=Depends(get_current_user),session:AsyncSession=Depends(get_db_session)):
 if p.refresh_token: await UserModel.revoke_refresh_token(session,p.refresh_token,commit=True)
 return {"success":True}
